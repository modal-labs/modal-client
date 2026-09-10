/**
 * Counts down to giving something back once nothing is using it.
 *
 * What counts as "in use", and what is given back, is the owner's business: one
 * owner keeps a connection while operations are in flight, another keeps a
 * stream while a read has it. This owns only the countdown.
 *
 * @internal
 */
export class IdleCountdown {
  #timer: ReturnType<typeof globalThis.setTimeout> | undefined;

  /** Ends the countdown in progress. */
  stop(): void {
    if (this.#timer !== undefined) {
      globalThis.clearTimeout(this.#timer);
      this.#timer = undefined;
    }
  }

  /**
   * Starts the countdown again, calling `release` when it elapses. A timeout of
   * zero or less leaves it stopped.
   */
  arm(timeoutMs: number, release: () => void): void {
    this.stop();
    if (timeoutMs <= 0) {
      return;
    }
    this.#timer = globalThis.setTimeout(() => {
      this.#timer = undefined;
      release();
    }, timeoutMs);
    // A pending release must not hold the process open on its own.
    this.#timer.unref?.();
  }
}
