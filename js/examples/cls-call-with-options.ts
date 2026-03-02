// This example calls a Modal Cls defined in `libmodal_test_support.py`,
// and overrides the default options.

import { ModalClient } from "modal";

const modal = new ModalClient();

const cls = await modal.cls.fromName(
  "libmodal-test-support",
  "EchoClsParametrized",
);
const instance = await cls.instance();
const method = instance.method("echo_env_var");

const instanceWithOptions = await cls
  .withOptions({
    secrets: [
      await modal.secrets.fromObject({ SECRET_MESSAGE: "hello, Secret" }),
    ],
  })
  .withConcurrency({ maxInputs: 1 })
  .instance();
const methodWithOptions = instanceWithOptions.method("echo_env_var");

// Call the Cls function, without the Secret being set.
console.log(await method.remote(["SECRET_MESSAGE"]));

// Call the Cls function with overrides, and confirm that the Secret is set.
console.log(await methodWithOptions.remote(["SECRET_MESSAGE"]));
