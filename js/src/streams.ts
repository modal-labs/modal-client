/**
 * Wrapper around `ReadableStream` with convenience functions.
 *
 * The Stream API is a modern standard for asynchronous data streams across
 * network and process boundaries. It allows you to read data in chunks, pipe
 * and transform it, and handle backpressure.
 *
 * This wrapper adds some extra functions like `.readText()` to read the entire
 * stream as a string, or `readBytes()` to read binary data.
 *
 * Background: https://developer.mozilla.org/en-US/docs/Web/API/Streams_API
 */
export interface ModalReadStream<R = any> extends ReadableStream<R> {
  /** Read the entire stream as a string. */
  readText(): Promise<string>;

  /** Read the entire stream as a byte array. */
  readBytes(): Promise<Uint8Array>;
}

/**
 * Wrapper around `WritableStream` with convenience functions.
 *
 * The Stream API is a modern standard for asynchronous data streams across
 * network and process boundaries. It allows you to read data in chunks, pipe
 * and transform it, and handle backpressure.
 *
 * This wrapper adds some extra functions like `.writeText()` to write a string
 * to the stream, or `writeBytes()` to write binary data.
 *
 * Background: https://developer.mozilla.org/en-US/docs/Web/API/Streams_API
 */
export interface ModalWriteStream<R = any> extends WritableStream<R> {
  /** Write a string to the stream. Only if this is a text stream. */
  writeText(text: string): Promise<void>;

  /** Write a byte array to the stream. Only if this is a byte stream. */
  writeBytes(bytes: Uint8Array): Promise<void>;
}

export function toModalReadStream<R extends string | Uint8Array = any>(
  stream: ReadableStream<R>,
): ModalReadStream<R> {
  return Object.assign(stream, readMixin);
}

export function toModalWriteStream<R extends string | Uint8Array = any>(
  stream: WritableStream<R>,
): ModalWriteStream<R> {
  return Object.assign(stream, writeMixin);
}

const readMixin = {
  async readText<R extends string | Uint8Array>(
    this: ReadableStream<R>,
  ): Promise<string> {
    const reader = this.getReader();
    try {
      const decoder = new TextDecoder("utf-8"); // used if binary
      const chunks: string[] = [];
      while (true) {
        const { value, done } = await reader.read();
        if (value) {
          if (typeof value === "string") chunks.push(value);
          else {
            chunks.push(decoder.decode(value, { stream: true }));
          }
        }
        if (done) {
          chunks.push(decoder.decode(undefined, { stream: false })); // may be empty
          break;
        }
      }
      return chunks.join("");
    } finally {
      reader.releaseLock();
    }
  },

  async readBytes<R extends string | Uint8Array>(
    this: ReadableStream<R>,
  ): Promise<Uint8Array> {
    const chunks: Uint8Array[] = [];
    const reader = this.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (value) {
          if (typeof value === "string") {
            chunks.push(new TextEncoder().encode(value));
          } else {
            chunks.push(value);
          }
        }
        if (done) break;
      }
    } finally {
      reader.releaseLock();
    }

    let totalLength = 0;
    for (const chunk of chunks) {
      totalLength += chunk.length;
    }
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result;
  },
};

const writeMixin = {
  async writeText<R extends string | Uint8Array>(
    this: WritableStream<R>,
    text: string,
  ): Promise<void> {
    const writer = this.getWriter();
    try {
      // Cast to R so TS is happy; underlying sink must accept strings
      await writer.write(text as unknown as R);
    } finally {
      writer.releaseLock();
    }
  },

  async writeBytes<R extends string | Uint8Array>(
    this: WritableStream<R>,
    bytes: Uint8Array,
  ): Promise<void> {
    const writer = this.getWriter();
    try {
      // Cast to R so TS is happy; underlying sink must accept Uint8Array
      await writer.write(bytes as unknown as R);
    } finally {
      writer.releaseLock();
    }
  },
};

/**
 * Construct a ReadableStream from an iterator.
 * If the stream is canceled, we signal the iterator via return() to stop
 * consumption and allow the source to clean up promptly.
 */
export function streamConsumingIter<R extends string | Uint8Array = Uint8Array>(
  iterable: AsyncIterable<R>,
  onCancel?: () => void,
): ReadableStream<R> {
  const iter = iterable[Symbol.asyncIterator]();
  return new ReadableStream<R>(
    {
      async pull(controller) {
        const { done, value } = await iter.next();
        if (value) {
          controller.enqueue(value);
        }
        if (done) {
          controller.close();
        }
      },
      async cancel() {
        try {
          onCancel?.();
        } finally {
          // Propagate cancellation upstream and run source cleanup.
          // return() is optional on AsyncIterator, so guard before calling.
          if (typeof iter.return === "function") {
            await iter.return();
          }
        }
      },
    },
    // The high water mark is set to zero intentionally to not run any
    // background buffering of data, since that would eagerly start streams etc.
    // If users want a buffered stream they can add buffing on top of the returned
    // streams instead.
    new CountQueuingStrategy({ highWaterMark: 0 }),
  );
}

/**
 * Decode a byte stream to text, holding partial characters across chunk
 * boundaries.
 *
 * This is used instead of built in decoder streams to avoid eager fetching
 * of data which would keep streams open unnecessarly.
 */
export async function* decodeTextIter(
  source: AsyncIterable<Uint8Array>,
): AsyncIterable<string> {
  const decoder = new TextDecoder();
  for await (const chunk of source) {
    const text = decoder.decode(chunk, { stream: true });
    if (text) {
      yield text;
    }
  }
  const rest = decoder.decode();
  if (rest) {
    yield rest;
  }
}
