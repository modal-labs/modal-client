import { expect, test } from "vitest";
import { ModalClient } from "modal";

declare const __MODAL_SDK_VERSION__: string;

test("VersionConstantFormat", () => {
  expect(__MODAL_SDK_VERSION__).toMatch(/^\d+\.\d+\.\d+(-dev\.\d+)?$/);
});

test("ClientVersion", () => {
  const client = new ModalClient();
  expect(client.version()).toMatch(/^\d+\.\d+\.\d+(-dev\.\d+)?$/);
  expect(client.version()).toBe(__MODAL_SDK_VERSION__);
  client.close();
});
