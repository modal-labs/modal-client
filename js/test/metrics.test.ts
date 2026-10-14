import {
  Function_,
  InvalidError,
  type FunctionMetricsExportParams,
} from "modal";
import { ClientError, Status } from "nice-grpc";
import { afterEach, expect, test, vi } from "vitest";
import {
  FunctionGetMetricsRequest,
  FunctionGetMetricsResponse,
  MetricsGetInfoRequest,
  MetricsGetInfoResponse,
  MetricsGetInfoRequest_MetricsTargetType,
} from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";

const start = new Date("2026-10-01T00:00:00Z");
const end = new Date("2026-10-01T01:00:00Z");
afterEach(() => vi.useRealTimers());

test("export passes options and preserves effective bounds, nulls, and zeroes", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const since = new Date(start.getTime() + 30000);
  const until = new Date(end.getTime() + 30000);
  mockCpClient.handleUnary("FunctionGetMetrics", (value) => {
    expect(value).toMatchObject({
      functionId: "fu-test",
      since,
      until,
      bucketSecs: 60,
      groups: ["calls"],
      rollup: true,
    });
    return FunctionGetMetricsResponse.create({
      since: start,
      until: end,
      bucketSecs: 60,
      series: [
        {
          name: "successful_calls_count",
          unit: "calls",
          description: "Successful inputs.",
          points: [
            { timestamp: start, value: 0 },
            { timestamp: new Date(start.getTime() + 60000) },
          ],
        },
        {
          name: "execution_time_p50",
          unit: "seconds",
          points: [{ timestamp: start, value: 1.5 }],
        },
      ],
    });
  });
  const result = await new Function_(mockClient, "fu-test").metrics.export({
    since,
    until,
    bucketSizeSecs: 60,
    groups: ["calls"],
    allVariants: true,
  });
  expect(result).toMatchObject({
    objectId: "fu-test",
    since: start,
    until: end,
    bucketSizeSecs: 60,
    pointCount: 3,
  });
  expect(result.series.successful_calls_count).toEqual({
    name: "successful_calls_count",
    unit: "calls",
    aggregation: "count",
    description: "Successful inputs.",
    points: [
      { time: start, value: 0 },
      { time: new Date(start.getTime() + 60000), value: null },
    ],
  });
  expect(result.series.execution_time_p50.aggregation).toBe("p50");
  mockCpClient.assertExhausted();
});

test.each([false, true])(
  "export defaults with explicit until=%s",
  async (explicitUntil) => {
    vi.useFakeTimers();
    vi.setSystemTime(end);
    const { mockClient, mockCpClient } = createMockModalClients();
    mockCpClient.handleUnary("FunctionGetMetrics", (value) => {
      const request = value as FunctionGetMetricsRequest;
      expect(request.since).toEqual(start);
      expect(request.until).toEqual(end);
      expect(request.groups).toEqual([]);
      expect(request.rollup).toBe(false);
      expect(request.bucketSecs).toBeUndefined();
      return FunctionGetMetricsResponse.create({
        since: start,
        until: end,
        bucketSecs: 60,
      });
    });
    const result = await new Function_(mockClient, "fu-test").metrics.export(
      explicitUntil ? { until: end } : {},
    );
    expect(result.series).toEqual({});
    expect(result.pointCount).toBe(0);
    mockCpClient.assertExhausted();
  },
);

test.each([undefined, ["future"]])(
  "schema forwards groups %s and merges rows without fetching samples",
  async (groups) => {
    const { mockClient, mockCpClient } = createMockModalClients();
    mockCpClient.handleUnary("MetricsGetInfo", (value) => {
      const request = value as MetricsGetInfoRequest;
      expect(request.targetType).toBe(
        MetricsGetInfoRequest_MetricsTargetType.METRICS_TARGET_TYPE_FUNCTION,
      );
      expect(request.groups).toEqual(groups ?? []);
      expect(request.until!.getTime() - request.since!.getTime()).toBe(3600000);
      return MetricsGetInfoResponse.create({
        groups: [
          {
            name: "future",
            displayName: "future_{a,b}_avg",
            metrics: [
              { name: "future_a_avg", unit: "widgets", description: "A" },
              { name: "future_b_avg", unit: "widgets", description: "B" },
            ],
          },
          {
            name: "future",
            metrics: [
              { name: "future_c_count", unit: "widgets", description: "C" },
            ],
          },
          {
            name: "__proto__",
            metrics: [{ name: "special", unit: "widgets" }],
          },
        ],
      });
    });
    const result = await new Function_(mockClient, "fu-test").metrics.schema({
      groups,
    });
    expect(Object.keys(result)).toEqual(["future", "__proto__"]);
    expect(result.future).toEqual({
      name: "future",
      definitions: [
        { name: "future_a_avg", unit: "widgets", description: "A" },
        { name: "future_b_avg", unit: "widgets", description: "B" },
        { name: "future_c_count", unit: "widgets", description: "C" },
      ],
    });
    expect(result["__proto__"].name).toBe("__proto__");
    mockCpClient.assertExhausted();
  },
);

test.each<[FunctionMetricsExportParams, string]>([
  [{ since: new Date(NaN) }, "valid dates"],
  [{ until: new Date(NaN) }, "valid dates"],
  [{ since: start, until: start }, "before"],
  [{ since: new Date(-1), until: end }, "Unix epoch"],
  [
    { since: start, until: new Date(start.getTime() + 31 * 86400000 + 1) },
    "31 days",
  ],
  ...[0, -1, 59, 60.5, NaN, Infinity, 2 ** 32].map(
    (bucketSizeSecs) =>
      [{ bucketSizeSecs }, "whole number"] as [
        FunctionMetricsExportParams,
        string,
      ],
  ),
  [
    {
      since: start,
      until: new Date(start.getTime() + 501 * 60000),
      bucketSizeSecs: 60,
    },
    "500 points",
  ],
  [{ groups: [""] }, "nonempty"],
])("export rejects invalid options %j", async (params, message) => {
  const { mockClient } = createMockModalClients();
  const result = new Function_(mockClient, "fu-test").metrics.export(params);
  await expect(result).rejects.toThrow(InvalidError);
  await expect(result).rejects.toThrow(message);
});

test.each(["schema", "export"] as const)(
  "%s propagates server validation errors",
  async (method) => {
    const { mockClient, mockCpClient } = createMockModalClients();
    const error = new ClientError(
      "/Metrics",
      Status.INVALID_ARGUMENT,
      "Invalid metrics groups: unknown",
    );
    mockCpClient.handleUnary(
      method === "schema" ? "MetricsGetInfo" : "FunctionGetMetrics",
      () => {
        throw error;
      },
    );
    await expect(
      new Function_(mockClient, "fu-test").metrics[method]({
        groups: ["unknown"],
      }),
    ).rejects.toBe(error);
    mockCpClient.assertExhausted();
  },
);

test("accepts exactly 31 days and 500 aligned points", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  for (const params of [
    { since: start, until: new Date(start.getTime() + 31 * 86400000) },
    {
      since: start,
      until: new Date(start.getTime() + 500 * 60000),
      bucketSizeSecs: 60,
    },
  ]) {
    mockCpClient.handleUnary("FunctionGetMetrics", () =>
      FunctionGetMetricsResponse.create({
        since: start,
        until: end,
        bucketSecs: 60,
      }),
    );
    await new Function_(mockClient, "fu-test").metrics.export(params);
  }
  mockCpClient.assertExhausted();
});
