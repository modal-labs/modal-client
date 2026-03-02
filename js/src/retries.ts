/** Retry policy configuration for a Modal Function/Cls. */
export class Retries {
  readonly maxRetries: number;
  readonly backoffCoefficient: number;
  readonly initialDelayMs: number;
  readonly maxDelayMs: number;

  constructor(params: {
    maxRetries: number;
    backoffCoefficient?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
  }) {
    const {
      maxRetries,
      backoffCoefficient = 2.0,
      initialDelayMs = 1000,
      maxDelayMs = 60_000,
    } = params;

    if (maxRetries < 0 || maxRetries > 10) {
      throw new Error(
        `Invalid maxRetries: ${maxRetries}. Must be between 0 and 10.`,
      );
    }

    if (backoffCoefficient < 1.0 || backoffCoefficient > 10.0) {
      throw new Error(
        `Invalid backoffCoefficient: ${backoffCoefficient}. Must be between 1.0 and 10.0`,
      );
    }

    if (initialDelayMs < 0 || initialDelayMs > 60_000) {
      throw new Error(
        `Invalid initialDelayMs: ${initialDelayMs}. Must be between 0 and 60000 ms.`,
      );
    }

    if (maxDelayMs < 1_000 || maxDelayMs > 60_000) {
      throw new Error(
        `Invalid maxDelayMs: ${maxDelayMs}. Must be between 1000 and 60000 ms.`,
      );
    }

    this.maxRetries = maxRetries;
    this.backoffCoefficient = backoffCoefficient;
    this.initialDelayMs = initialDelayMs;
    this.maxDelayMs = maxDelayMs;
  }
}

export function parseRetries(
  retries: number | Retries | undefined,
): Retries | undefined {
  if (retries === undefined) return undefined;
  if (typeof retries === "number") {
    if (!Number.isInteger(retries) || retries < 0 || retries > 10) {
      throw new Error(
        `Retries parameter must be an integer between 0 and 10. Found: ${retries}`,
      );
    }
    return new Retries({
      maxRetries: retries,
      backoffCoefficient: 1.0,
      initialDelayMs: 1000,
    });
  }
  if (retries instanceof Retries) return retries;
  throw new Error(
    `Retries parameter must be an integer or instance of Retries. Found: ${typeof retries}.`,
  );
}
