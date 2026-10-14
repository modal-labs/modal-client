import {
  FunctionGetMetricsRequest,
  MetricsGetInfoRequest,
  MetricsGetInfoRequest_MetricsTargetType,
} from "../proto/modal_proto/api";
import type { ModalClient } from "./client";
import { InvalidError } from "./errors";

/** An observation at a UTC-aligned bucket start. */
export interface TimeSeriesPoint {
  time: Date;
  /** Missing observations are null; observed zeroes remain zero. */
  value: number | null;
}

/** An individual metric series with points ordered by time. */
export interface MetricSeries {
  name: string;
  unit: string;
  aggregation: string;
  description: string;
  points: TimeSeriesPoint[];
}

/** The name, unit, and meaning of an exportable series. */
export interface MetricDefinition {
  name: string;
  unit: string;
  description: string;
}

/** A metrics group and its individual series definitions. */
export interface MetricsSchema {
  name: string;
  definitions: MetricDefinition[];
}

/** Exported metrics over the effective, UTC-aligned [since, until) range. */
export interface ObjectMetrics {
  objectId: string;
  since: Date;
  until: Date;
  bucketSizeSecs: number;
  /** Total points across series, including missing observations. */
  pointCount: number;
  series: Record<string, MetricSeries>;
}

/** Options for discovering metric definitions. */
export interface MetricsSchemaParams {
  /** Omit or pass an empty array to select all groups. */
  groups?: string[];
}

/** Options for exporting Function metric samples. */
export interface FunctionMetricsExportParams extends MetricsSchemaParams {
  /** Defaults to one hour before until. */
  since?: Date;
  /** Defaults to the current time. */
  until?: Date;
  /** Bucket width in whole seconds. */
  bucketSizeSecs?: number;
  /** Include the root Function and all its variants. Defaults to false. */
  allVariants?: boolean;
}

function metricsRequest(
  params: FunctionMetricsExportParams,
): FunctionGetMetricsRequest {
  const until = params.until ?? new Date();
  const since = params.since ?? new Date(until.getTime() - 60 * 60 * 1000);
  if (!Number.isFinite(since.getTime()) || !Number.isFinite(until.getTime())) {
    throw new InvalidError("`since` and `until` must be valid dates.");
  }
  if (since >= until) {
    throw new InvalidError("`since` must be before `until`.");
  }
  if (since.getTime() < 0) {
    throw new InvalidError("`since` must be at or after the Unix epoch.");
  }
  if (until.getTime() - since.getTime() > 31 * 24 * 60 * 60 * 1000) {
    throw new InvalidError("Metrics time range cannot exceed 31 days.");
  }
  if (
    params.groups !== undefined &&
    (!Array.isArray(params.groups) ||
      params.groups.some((g) => typeof g !== "string" || !g))
  ) {
    throw new InvalidError(
      "`groups` must be an array of nonempty group names.",
    );
  }
  const request = FunctionGetMetricsRequest.create({
    since,
    until,
    groups: params.groups ?? [],
    rollup: params.allVariants ?? false,
  });
  if (params.bucketSizeSecs !== undefined) {
    const seconds = params.bucketSizeSecs;
    if (!Number.isInteger(seconds) || seconds < 60 || seconds > 2 ** 32 - 1) {
      throw new InvalidError(
        "`bucketSizeSecs` must be a whole number of seconds, at least one minute.",
      );
    }
    if (
      Math.floor(until.getTime() / (params.bucketSizeSecs * 1000)) -
        Math.floor(since.getTime() / (params.bucketSizeSecs * 1000)) >
      500
    ) {
      throw new InvalidError(
        "Metrics are limited to 500 points per series; increase `bucketSizeSecs`.",
      );
    }
    request.bucketSecs = seconds;
  }
  return request;
}

/** Namespace for Function metric samples and definitions. */
export class FunctionMetricsManager {
  #client: ModalClient;
  #functionId: string;

  /** @hidden */
  constructor(client: ModalClient, functionId: string) {
    this.#client = client;
    this.#functionId = functionId;
  }

  /**
   * Fetch metric samples, defaulting to all groups over the last hour.
   *
   * The maximum range is 31 days, with at most 500 points per series. Omit
   * bucketSizeSecs to let Modal select a supported width targeting approximately
   * 100 buckets. Bounds round down to UTC bucket boundaries: data before since
   * may be included, and the partial bucket at until is excluded.
   * Missing observations have value null. Use schema() to discover definitions.
   *
   * @example
   * ```typescript
   * const metrics = await fn.metrics.export({
   *   bucketSizeSecs: 5 * 60,
   *   groups: ["calls"],
   * });
   * for (const point of metrics.series["successful_calls_count"].points) {
   *   console.log(point.time, point.value);
   * }
   * ```
   */
  async export(
    params: FunctionMetricsExportParams = {},
  ): Promise<ObjectMetrics> {
    const request = metricsRequest(params);
    request.functionId = this.#functionId;
    const response = await this.#client.cpClient.functionGetMetrics(request);
    const series: Record<string, MetricSeries> = Object.fromEntries(
      response.series.map((item) => [
        item.name,
        {
          name: item.name,
          unit: item.unit,
          aggregation: item.name.slice(item.name.lastIndexOf("_") + 1),
          description: item.description,
          points: item.points.map((point) => ({
            time: point.timestamp ?? new Date(0),
            value: point.value ?? null,
          })),
        },
      ]),
    );
    return {
      objectId: this.#functionId,
      since: response.since ?? new Date(0),
      until: response.until ?? new Date(0),
      bucketSizeSecs: response.bucketSecs,
      pointCount: Object.values(series).reduce(
        (count, item) => count + item.points.length,
        0,
      ),
      series,
    };
  }

  /**
   * Return metric definitions keyed by group name, without fetching samples.
   * Omit groups to return all groups. Each group's definitions include all its
   * individual series.
   */
  async schema(
    params: MetricsSchemaParams = {},
  ): Promise<Record<string, MetricsSchema>> {
    const window = metricsRequest(params);
    const response = await this.#client.cpClient.metricsGetInfo(
      MetricsGetInfoRequest.create({
        targetType:
          MetricsGetInfoRequest_MetricsTargetType.METRICS_TARGET_TYPE_FUNCTION,
        since: window.since,
        until: window.until,
        groups: window.groups,
      }),
    );
    const groups = new Map<string, MetricsSchema>();
    for (const row of response.groups) {
      let group = groups.get(row.name);
      if (!group) {
        group = { name: row.name, definitions: [] };
        groups.set(row.name, group);
      }
      group.definitions.push(
        ...row.metrics.map((metric) => ({
          name: metric.name,
          unit: metric.unit,
          description: metric.description,
        })),
      );
    }
    return Object.fromEntries(groups);
  }
}
