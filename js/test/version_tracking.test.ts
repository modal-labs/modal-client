import { expect, test } from "vitest";
import { ModalClient } from "modal";

// Consistency between the checked-in version and package.json is enforced by
// the `inv lint-versions` linter, not here. This only checks the client
// surfaces a well-formed version at runtime.
test("ClientVersion", () => {
  const client = new ModalClient();
  expect(client.version()).toMatch(/^\d+\.\d+\.\d+(-dev\.\d+)?$/);
  client.close();
});
