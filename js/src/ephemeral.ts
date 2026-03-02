export const ephemeralObjectHeartbeatSleep = 300_000; // 300 seconds

export type HeartbeatFunction = () => Promise<any>;

export class EphemeralHeartbeatManager {
  private readonly heartbeatFn: HeartbeatFunction;
  private readonly abortController: AbortController;

  constructor(heartbeatFn: HeartbeatFunction) {
    this.heartbeatFn = heartbeatFn;
    this.abortController = new AbortController();

    this.start();
  }

  private start(): void {
    const signal = this.abortController.signal;
    (async () => {
      while (!signal.aborted) {
        await this.heartbeatFn();
        await Promise.race([
          new Promise((resolve) => {
            // unref so the heartbeat timer doesn't prevent the process from exiting
            setTimeout(resolve, ephemeralObjectHeartbeatSleep).unref();
          }),
          new Promise((resolve) => {
            signal.addEventListener("abort", resolve, { once: true });
          }),
        ]);
      }
    })();
  }

  stop(): void {
    this.abortController.abort();
  }
}
