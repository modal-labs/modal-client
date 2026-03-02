// This example calls a Function defined in `libmodal_test_support.py`.

import { ModalClient } from "modal";

const modal = new ModalClient();

const echo = await modal.functions.fromName(
  "libmodal-test-support",
  "echo_string",
);

// Spawn the Function with kwargs.
const functionCall = await echo.spawn([], { s: "Hello world!" });
const ret = await functionCall.get();
console.log(ret);
