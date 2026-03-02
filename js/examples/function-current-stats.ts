// Demonstrates how to get current statistics for a Modal Function.

import { ModalClient } from "modal";

const modal = new ModalClient();

const func = await modal.functions.fromName(
  "libmodal-test-support",
  "echo_string",
);

const stats = await func.getCurrentStats();

console.log("Function Statistics:");
console.log(`  Backlog: ${stats.backlog} inputs`);
console.log(`  Total Runners: ${stats.numTotalRunners} containers`);
