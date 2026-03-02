import { ClientMiddlewareCall, CallOptions } from "nice-grpc";
import { ModalClient } from "modal";
import { expect, test } from "vitest";

test("ModalClient with custom middleware", async () => {
  let firstCalled = false;
  let secondCalled = false;
  let firstMethod = "";
  let secondMethod = "";

  const firstMiddleware = async function* <Request, Response>(
    call: ClientMiddlewareCall<Request, Response>,
    options: CallOptions,
  ) {
    firstCalled = true;
    firstMethod = call.method.path;
    return yield* call.next(call.request, options);
  };

  const secondMiddleware = async function* <Request, Response>(
    call: ClientMiddlewareCall<Request, Response>,
    options: CallOptions,
  ) {
    secondCalled = true;
    secondMethod = call.method.path;
    return yield* call.next(call.request, options);
  };

  const mc = new ModalClient({
    grpcMiddleware: [firstMiddleware, secondMiddleware],
  });

  try {
    await mc.functions.fromName("libmodal-test-support", "non-existent");
  } catch (_err) {
    // Don't care about success here, just need the RPC to be made
  } finally {
    mc.close();
  }

  expect(firstCalled).toBe(true);
  expect(firstMethod).toContain("ModalClient/");
  expect(secondCalled).toBe(true);
  expect(secondMethod).toContain("ModalClient/");
});
