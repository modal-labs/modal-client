// This example calls a Modal Cls defined in `libmodal_test_support.py` and
// overrides its routing region, so that inputs/outputs are routed through a specific
// region at invocation time. Modal Functions can be similarly rerouted.

import { ModalClient } from "modal";

const modal = new ModalClient();

const cls = await modal.cls.fromName(
  "libmodal-test-support",
  "EchoClsInputPlane",
);

// Override the class's default routing region so that inputs/outputs are routed
// through us-west.
const instance = await cls.withOptions({ routingRegion: "us-west" }).instance();
const method = instance.method("echo_string");

console.log(await method.remote(["hello"]));
