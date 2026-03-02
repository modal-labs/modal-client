import { App, Proxy } from "modal";
import { expect, test } from "vitest";

test("CreateSandboxWithProxy", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const proxy = await Proxy.fromName("libmodal-test-proxy", {
    environment: "libmodal",
  });
  expect(proxy.proxyId).toBeTruthy();
  expect(proxy.proxyId).toMatch(/^pr-/);

  const sb = await app.createSandbox(image, {
    proxy,
    command: ["echo", "hello, sandbox with proxy"],
  });
  expect(sb.sandboxId).toBeTruthy();
  expect(await sb.terminate({ wait: true })).toBe(137);
});

test("ProxyNotFound", async () => {
  await expect(Proxy.fromName("non-existent-proxy-name")).rejects.toThrow(
    "Proxy 'non-existent-proxy-name' not found",
  );
});
