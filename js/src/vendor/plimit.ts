/*
 * This pLimit implementation is adapted from p-limit.
 *
 * MIT License
 * Copyright (c) Sindre Sorhus (https://sindresorhus.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

type QueueNode<T> = {
  value: T;
  next?: QueueNode<T>;
};

class FifoQueue<T> {
  #head?: QueueNode<T>;
  #tail?: QueueNode<T>;
  #size = 0;

  get size(): number {
    return this.#size;
  }

  enqueue(value: T): void {
    const node = { value };

    if (this.#tail === undefined) {
      this.#head = node;
    } else {
      this.#tail.next = node;
    }

    this.#tail = node;
    this.#size++;
  }

  dequeue(): T | undefined {
    const node = this.#head;
    if (node === undefined) {
      return undefined;
    }

    this.#head = node.next;
    this.#size--;
    if (this.#head === undefined) {
      this.#tail = undefined;
    }

    return node.value;
  }

  clear(): void {
    this.#head = undefined;
    this.#tail = undefined;
    this.#size = 0;
  }
}

type MaybePromise<T> = T | PromiseLike<T>;

export interface Limit {
  <Arguments extends unknown[], ReturnType>(
    function_: (...arguments_: Arguments) => MaybePromise<ReturnType>,
    ...arguments_: Arguments
  ): Promise<Awaited<ReturnType>>;

  readonly activeCount: number;
  readonly pendingCount: number;
  concurrency: number;

  clearQueue(): void;

  map<Element, ReturnType>(
    iterable: Iterable<Element>,
    mapper: (element: Element, index: number) => MaybePromise<ReturnType>,
  ): Promise<Awaited<ReturnType>[]>;
}

type QueueItem = {
  run(): void;
};

/**
 * Limits the number of concurrently running promise-returning functions.
 *
 * @internal
 * @hidden
 */
export function pLimit(concurrency: number): Limit {
  validateConcurrency(concurrency);

  const queue = new FifoQueue<QueueItem>();
  let activeCount = 0;

  const resumeNext = (): void => {
    if (activeCount < concurrency) {
      const item = queue.dequeue();
      if (item !== undefined) {
        activeCount++;
        item.run();
      }
    }
  };

  const drainQueue = (): void => {
    while (activeCount < concurrency) {
      if (queue.size === 0) {
        return;
      }
      resumeNext();
    }
  };

  const limit = (<Arguments extends unknown[], ReturnType>(
    function_: (...arguments_: Arguments) => MaybePromise<ReturnType>,
    ...arguments_: Arguments
  ): Promise<Awaited<ReturnType>> =>
    new Promise<Awaited<ReturnType>>((resolve) => {
      const item: QueueItem = { run: () => {} };

      new Promise<void>((start) => {
        item.run = start;
        queue.enqueue(item);
      }).then(async () => {
        const result = (async (): Promise<Awaited<ReturnType>> =>
          await function_(...arguments_))();
        resolve(result);

        try {
          await result;
        } catch {
          // The rejection is propagated through the promise returned by limit.
        }

        activeCount--;
        resumeNext();
      });

      resumeNext();
    })) as Limit;

  Object.defineProperties(limit, {
    activeCount: {
      get: () => activeCount,
    },
    pendingCount: {
      get: () => queue.size,
    },
    clearQueue: {
      value() {
        queue.clear();
      },
    },
    concurrency: {
      get: () => concurrency,
      set(newConcurrency: number) {
        validateConcurrency(newConcurrency);
        concurrency = newConcurrency;
        queueMicrotask(drainQueue);
      },
    },
    map: {
      async value<Element, ReturnType>(
        iterable: Iterable<Element>,
        mapper: (element: Element, index: number) => MaybePromise<ReturnType>,
      ): Promise<Awaited<ReturnType>[]> {
        const promises = Array.from(iterable, (element, index) =>
          limit(mapper, element, index),
        );
        return Promise.all(promises);
      },
    },
  });

  return limit;
}

function validateConcurrency(concurrency: number): void {
  if (
    !(
      (Number.isInteger(concurrency) ||
        concurrency === Number.POSITIVE_INFINITY) &&
      concurrency > 0
    )
  ) {
    throw new TypeError("Expected concurrency to be a number from 1 and up");
  }
}
