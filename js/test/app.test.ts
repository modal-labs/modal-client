import { App, InvalidError } from "modal";
import { expect, test } from "vitest";
import {
  AppGetTimeRangeStatsRequest,
  AppGetTimeRangeStatsResponse,
  AppState,
  AppStopSource,
} from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";

test("AppStop", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", () => ({
    appId: "ap-test-123",
  }));

  mock.handleUnary("/AppStop", (req: any) => {
    expect(req.appId).toBe("ap-test-123");
    expect(req.source).toBe(AppStopSource.APP_STOP_SOURCE_PYTHON_CLIENT);
    return {};
  });

  const app = await mc.apps.fromName("libmodal-test");
  await app.stop();
  mock.assertExhausted();
});

test("AppStop without an App ID", async () => {
  const { mockClient: mc } = createMockModalClients();
  const app = new App("", undefined, undefined, mc);
  await expect(app.stop()).rejects.toThrow(InvalidError);
});

test("AppStop without a client", async () => {
  const app = new App("ap-test-123");
  await expect(app.stop()).rejects.toThrow(InvalidError);
});

test.each([undefined, 0, 4])(
  "AppStats maps response with deploymentCount %s",
  async (deploymentCount) => {
    const { mockClient, mockCpClient } = createMockModalClients();
    const since = new Date("2026-09-01T00:00:00Z");
    const until = new Date("2026-09-01T01:00:00Z");
    mockCpClient.handleUnary("AppGetTimeRangeStats", (request) => {
      expect(request).toEqual(
        AppGetTimeRangeStatsRequest.create({
          appId: "ap-test-123",
          since,
          until,
        }),
      );
      return AppGetTimeRangeStatsResponse.create({
        since,
        until,
        functions: {
          total: {
            calls: { total: 10, succeeded: 7, failed: 3 },
            containers: { containerErrors: 2, containerActiveCount: 5 },
          },
          byName: {
            embed: {
              objectId: "fu-embed",
              counts: {
                calls: { total: 6, succeeded: 5, failed: 1 },
                containers: { containerErrors: 1, containerActiveCount: 3 },
              },
            },
          },
        },
        servers: {
          total: {
            requests: { total: 20, ok: 16, failed: 4 },
            containers: { containerErrors: 3, containerActiveCount: 8 },
          },
          byName: {
            web: {
              objectId: "fu-web",
              counts: {
                requests: { total: 9, ok: 7, failed: 2 },
                containers: { containerErrors: 1, containerActiveCount: 4 },
              },
            },
          },
        },
        sandboxes: { total: 12, finished: 8, live: 4 },
        lifecycle: {
          appState: AppState.APP_STATE_DEPLOYED,
          version: 2,
          createdAt: 100,
          createdBy: "creator",
          deployedAt: 200,
          deployedBy: "deployer",
          stoppedAt: 300,
          stoppedBy: "stopper",
        },
        deploymentCount,
      });
    });
    expect(
      await new App("ap-test-123", undefined, undefined, mockClient).stats({
        since,
        until,
      }),
    ).toEqual({
      since,
      until,
      functions: {
        total: {
          calls: { total: 10, succeeded: 7, failed: 3 },
          containers: { errors: 2, total: 5 },
        },
        byName: {
          embed: {
            objectId: "fu-embed",
            counts: {
              calls: { total: 6, succeeded: 5, failed: 1 },
              containers: { errors: 1, total: 3 },
            },
          },
        },
      },
      servers: {
        total: {
          requests: { total: 20, ok: 16, failed: 4 },
          containers: { errors: 3, total: 8 },
        },
        byName: {
          web: {
            objectId: "fu-web",
            counts: {
              requests: { total: 9, ok: 7, failed: 2 },
              containers: { errors: 1, total: 4 },
            },
          },
        },
      },
      sandboxes: { total: 12, finished: 8, live: 4 },
      lifecycle: {
        state: "deployed",
        version: 2,
        createdAt: new Date(100000),
        createdBy: "creator",
        deployedAt: new Date(200000),
        deployedBy: "deployer",
        stoppedAt: new Date(300000),
        stoppedBy: "stopper",
      },
      deploymentCount,
    });
    mockCpClient.assertExhausted();
  },
);

test("AppStats defaults and empty results", async () => {
  for (const params of [{}, { until: new Date("2026-09-01T00:00:00Z") }]) {
    const { mockClient, mockCpClient } = createMockModalClients();
    const before = Date.now();
    mockCpClient.handleUnary("AppGetTimeRangeStats", (request) => {
      const req = request as AppGetTimeRangeStatsRequest;
      expect(req.appId).toBe("ap-test-123");
      expect(req.until!.getTime() - req.since!.getTime()).toBe(3600000);
      if (params.until) expect(req.until).toEqual(params.until);
      else {
        expect(req.until!.getTime()).toBeGreaterThanOrEqual(before);
        expect(req.until!.getTime()).toBeLessThanOrEqual(Date.now());
      }
      return AppGetTimeRangeStatsResponse.create({
        since: req.since,
        until: req.until,
      });
    });
    const result = await new App(
      "ap-test-123",
      undefined,
      undefined,
      mockClient,
    ).stats(params);
    expect(result.functions).toEqual({
      total: {
        calls: { total: 0, succeeded: 0, failed: 0 },
        containers: { errors: 0, total: 0 },
      },
      byName: {},
    });
    expect(result.servers).toEqual({
      total: {
        requests: { total: 0, ok: 0, failed: 0 },
        containers: { errors: 0, total: 0 },
      },
      byName: {},
    });
    expect(result.sandboxes).toEqual({ total: 0, finished: 0, live: 0 });
    expect(result.deploymentCount).toBeUndefined();
    mockCpClient.assertExhausted();
  }
});

test("AppStats rejects invalid ranges and propagates RPC errors", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const app = new App("ap-test-123", undefined, undefined, mockClient);
  const until = new Date("2026-09-01T00:00:00Z");
  for (const since of [until, new Date(until.getTime() + 1), new Date(NaN)]) {
    await expect(app.stats({ since, until })).rejects.toThrow(InvalidError);
  }
  await expect(app.stats({ until: new Date(NaN) })).rejects.toThrow(
    InvalidError,
  );
  mockCpClient.handleUnary("AppGetTimeRangeStats", () => {
    throw new Error("stats unavailable");
  });
  await expect(app.stats()).rejects.toThrow("stats unavailable");
  mockCpClient.assertExhausted();
});

test("AppStats without a client", async () => {
  await expect(new App("ap-test-123").stats()).rejects.toThrow(InvalidError);
});
