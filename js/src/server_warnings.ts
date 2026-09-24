import type { ClientMiddleware, Metadata } from "nice-grpc";
import type { Logger } from "./logger";

/**
 * Trailing metadata key the server uses to attach a non-fatal message to any
 * response, one percent-encoded entry per warning.
 * @internal
 */
export const SERVER_WARNING_HEADER = "x-modal-warning";

/** @internal */
export const SERVER_WARNING_REGISTRY_LIMIT = 2048;

/**
 * Log each warning in a response's trailer once per `seen` registry. The
 * server attaches some warnings to every RPC, so without deduplication a
 * long-running client would repeat them on every call and retry attempt.
 * @internal
 */
export function logServerWarnings(
  logger: Logger,
  trailer: Metadata,
  seen: Set<string>,
): void {
  const joined = trailer.get(SERVER_WARNING_HEADER);
  if (typeof joined !== "string") return;
  // Metadata joins repeated entries with ", "; each entry has its commas percent-encoded.
  for (const encoded of joined.split(",")) {
    let message: string;
    try {
      message = decodeURIComponent(encoded.trim());
    } catch {
      message = encoded.trim();
    }
    if (seen.size > SERVER_WARNING_REGISTRY_LIMIT) {
      // Start over instead of going quiet: overflow costs a repeat, never a hidden warning.
      seen.clear();
    }
    if (seen.has(message)) continue;
    seen.add(message);
    logger.warn(message);
  }
}

/**
 * Middleware that logs warnings the server attached to a response, including
 * failed ones. Each distinct message is logged once per middleware instance.
 * @internal
 */
export function serverWarningMiddleware(logger: Logger): ClientMiddleware {
  const seen = new Set<string>();
  return async function* serverWarningMiddleware(call, options) {
    const { onTrailer, ...restOptions } = options;
    return yield* call.next(call.request, {
      ...restOptions,
      onTrailer: (trailer: Metadata) => {
        logServerWarnings(logger, trailer, seen);
        onTrailer?.(trailer);
      },
    });
  };
}
