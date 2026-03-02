// This example demonstrates how to add custom telemetry and tracing to Modal API calls
// using gRPC middleware. It shows a simple custom middleware, but the same principle
// could also be used with e.g. OpenTelemetry for distributed tracing.

import { CallOptions, ClientMiddlewareCall } from "nice-grpc";
import { ModalClient } from "modal";

// telemetryMiddleware is a custom middleware that measures API call latency
// and logs method names with timing information.
async function* telemetryMiddleware<Request, Response>(
  call: ClientMiddlewareCall<Request, Response>,
  options: CallOptions,
) {
  const start = Date.now();

  try {
    const result = yield* call.next(call.request, options);
    const duration = Date.now() - start;
    console.log(
      `[TELEMETRY] method=${call.method.path} duration=${duration}ms status=success`,
    );
    // You could also send this data to your observability backend, etc.
    return result;
  } catch (err) {
    const duration = Date.now() - start;
    console.log(
      `[TELEMETRY] method=${call.method.path} duration=${duration}ms status=error`,
    );
    throw err;
  }
}

console.log(
  "Initializing Modal client with telemetry middleware. All API calls will be logged with timing information.",
);

const modal = new ModalClient({
  grpcMiddleware: [telemetryMiddleware],
});

const echo = await modal.functions.fromName(
  "libmodal-test-support",
  "echo_string",
);

const result = await echo.remote([], { s: "Hello from telemetry example!" });
console.log("Result:", result);

console.log("\nAll operations completed. See the telemetry logs above!");
