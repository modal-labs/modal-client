import { type ModalClient, isRetryableGrpc } from "./client";
import {
  AppCountLogsResponse,
  AppCountLogsRequest,
  AppCountLogsResponse_LogBucket,
  AppFetchLogsRequest,
  AppGetLogsRequest,
  FileDescriptor,
  type FunctionCallInfo,
  FunctionCallGetInfoRequest,
  type TaskLogsBatch,
  type TaskLogs,
} from "../proto/modal_proto/api";
import { pLimit } from "./vendor/plimit";
import { ClientError, Status } from "nice-grpc";

const tailLookbacksMs = [
  1000 * 60 * 60, // 1 hour
  1000 * 60 * 60 * 24, // 1 day
  1000 * 60 * 60 * 24 * 7, // 7 days
  1000 * 60 * 60 * 24 * 30, // 30 days
];

const FETCH_LIMIT = 20_000;
const INTERVAL_LOG_THRESHOLD = 2_000;
const MAX_FETCH_RANGE = 35 * 24 * 60 * 60 * 1000; // 35 days in milliseconds
// Maximum number of concurrent AppFetchLogs requests.
const MAX_CONCURRENT_FETCHES = 10;

// Maximum number of concurrent AppCountLogs requests (responses are small).
const MAX_CONCURRENT_COUNTS = 20;

const STREAM_POLL_INTERVAL_MS = 1_000;
const STREAM_RPC_TIMEOUT_SECONDS = 55;
const STREAM_DRAIN_TIMEOUT_SECONDS = 0.5;
const STREAM_RETRIES = 10;

// Maximum number of refinement iterations (each fires a parallel batch of AppCountLogs RPCs).
const MAX_REFINE_ITERATIONS = 3;

// Number of buckets for the initial count
const APPROX_INITIAL_BUCKETS = 100;

// Maximum number of fetch RPCs per fetch_logs call. Together with FETCH_LIMIT,
// this bounds total retrievable entries to MAX_FETCHES * FETCH_LIMIT = 10M.
const MAX_FETCHES = 500;
const FETCH_ERROR_MESSAGE =
  "Too many logs to fetch in time range. Consider narrowing the range or adding filters.";

// Predefined bucket sizes (in seconds)
// We pick the smallest size such that the total number of buckets stays <= APPROX_INITIAL_BUCKETS.
const BUCKET_SIZES_SECS = [
  2, 4, 6, 12, 20, 30, 60, 120, 180, 240, 300, 360, 600, 720, 900, 1200, 1800,
  3600, 7200, 10800, 14400, 28800, 43200, 86400,
];

export type LogSource = "system" | "stdout" | "stderr";

export interface LogEntry {
  readonly message: string;
  readonly source: LogSource;
  readonly timestamp: Date;
  readonly objectId: string;
  readonly contextIds: string[];
}

export interface FunctionLogFetchParams {
  /** Start of the UTC time range. */
  since: Date;
  /** End of the UTC time range. Defaults to the current time. */
  until?: Date;
  /** Filter by source: `stdout`, `stderr`, or `system`. */
  source?: LogSource;
  /** Filter by text contained in the log message. */
  searchText?: string;
}

export interface FunctionLogTailParams {
  /** Number of log entries to return. Defaults to 100. */
  entries?: number;
  /** Filter by source: `stdout`, `stderr`, or `system`. */
  source?: LogSource;
}

export interface FunctionCallLogTailParams {
  /** Number of log entries to return. Defaults to 100. */
  entries?: number;
  /** Filter by source: `stdout`, `stderr`, or `system`. */
  source?: LogSource;
}

type LogTailParams = FunctionCallLogTailParams | FunctionLogTailParams;

export interface FunctionLogStreamParams {
  /**
   * Number of milliseconds to wait between log entries before ending the
   * stream. By default, the stream blocks until it is interrupted.
   */
  timeoutMs?: number;
}

export interface FunctionCallLogStreamParams {
  /**
   * Number of milliseconds to wait between log entries before ending the
   * stream. By default, the stream blocks until it is interrupted.
   */
  timeoutMs?: number;
}

type LogStreamParams = FunctionCallLogStreamParams | FunctionLogStreamParams;

export interface FunctionCallLogFetchParams {
  /**
   * Start of the UTC time range. Defaults to the start of the FunctionCall.
   */
  since?: Date;
  /** End of the UTC time range. Defaults to the current time. */
  until?: Date;
  /** Filter by source: `stdout`, `stderr`, or `system`. */
  source?: LogSource;
  /** Filter by text contained in the log message. */
  searchText?: string;
}

interface FunctionLogs {
  fetch(params: FunctionLogFetchParams): AsyncIterable<LogEntry>;
  tail(params?: FunctionLogTailParams): AsyncIterable<LogEntry>;
  stream(params?: FunctionLogStreamParams): AsyncIterable<LogEntry>;
}

interface FunctionCallLogs extends Omit<FunctionLogs, "fetch"> {
  fetch(params?: FunctionCallLogFetchParams): AsyncIterable<LogEntry>;
  tail(params?: FunctionCallLogTailParams): AsyncIterable<LogEntry>;
  stream(params?: FunctionCallLogStreamParams): AsyncIterable<LogEntry>;
}

interface LogQueryParams {
  client: ModalClient;
  objectId: string;
  appId: string;
  functionId: string;
  functionCallId: string;
}

interface LogFilters {
  appId: string;
  objectId: string;
  functionId?: string;
  functionCallId?: string;
  taskId?: string;
  sandboxId?: string;
  searchText?: string;
  source: FileDescriptor;
}

interface LogRange {
  startMs: number;
  endMs: number;
  count: number;
}

interface RangeRefinement {
  index: number;
  range: LogRange;
  bucketSecs: number;
}

interface FetchInterval {
  startMs: number;
  endMs: number;
}

type StreamSignal =
  | { kind: "stop" }
  | { kind: "idle" }
  | { kind: "stopError"; error: unknown };

type StreamAttemptEvent =
  | { kind: "batch"; batch: TaskLogsBatch }
  | { kind: "eof" }
  | { kind: "streamError"; error: unknown }
  | StreamSignal;

type StreamRetryAction =
  | { kind: "reconnect" }
  | { kind: "fail"; error: unknown }
  | StreamSignal;

function contextIds(
  batch: TaskLogsBatch,
  item: TaskLogs,
  ObjectId: string,
): string[] {
  switch (ObjectId.slice(0, 2)) {
    case "fu":
      return [
        item.functionCallId,
        item.inputId || batch.inputId,
        item.containerId || batch.taskId,
      ].filter((id) => id);
    case "fc":
      return [
        item.inputId || batch.inputId,
        item.containerId || batch.taskId,
      ].filter((id) => id);
    default:
      return [];
  }
}

function sourceToFileDescriptor(source: LogSource | undefined): FileDescriptor {
  switch (source) {
    case "stdout":
      return FileDescriptor.FILE_DESCRIPTOR_STDOUT;
    case "stderr":
      return FileDescriptor.FILE_DESCRIPTOR_STDERR;
    case "system":
      return FileDescriptor.FILE_DESCRIPTOR_INFO;
    case undefined:
      return FileDescriptor.FILE_DESCRIPTOR_UNSPECIFIED;
    default:
      throw new Error(`Invalid log source: ${source}`);
  }
}

function fileDescriptorToSource(fd: FileDescriptor): LogSource {
  switch (fd) {
    case FileDescriptor.FILE_DESCRIPTOR_STDOUT:
      return "stdout";
    case FileDescriptor.FILE_DESCRIPTOR_STDERR:
      return "stderr";
    default:
      return "system";
  }
}

async function* tailLogs(
  client: ModalClient,
  n: number,
  filters: LogFilters,
): AsyncIterable<LogEntry> {
  if (n > FETCH_LIMIT) {
    throw new Error(`Cannot fetch more than ${FETCH_LIMIT} log entries.`);
  }
  if (n < 0) {
    throw new Error(`Cannot fetch a negative number of log entries.`);
  }

  const anchor = Date.now();
  for (const lookbackMs of tailLookbacksMs) {
    const lookbackSince = anchor - lookbackMs;
    const req = AppFetchLogsRequest.create({
      appId: filters.appId,
      since: new Date(lookbackSince),
      until: new Date(anchor),
      limit: n,
      source: filters.source,
      functionId: filters.functionId,
      functionCallId: filters.functionCallId,
      taskId: filters.taskId,
      sandboxId: filters.sandboxId,
      searchText: filters.searchText,
    });
    const resp = await client.cpClient.appFetchLogs(req);

    const totalItems = resp.batches.reduce(
      (acc, batch) => acc + batch.items.length,
      0,
    );
    if (
      totalItems >= n ||
      lookbackMs === tailLookbacksMs[tailLookbacksMs.length - 1]
    ) {
      for (const batch of resp.batches) {
        for (const item of batch.items) {
          yield {
            message: item.data,
            source: fileDescriptorToSource(item.fileDescriptor),
            timestamp: new Date(
              item.timestampNs && item.timestampNs !== "0"
                ? Number(BigInt(item.timestampNs) / 1_000_000n)
                : item.timestamp * 1_000,
            ),
            objectId: filters.objectId,
            contextIds: contextIds(batch, item, filters.objectId),
          };
        }
      }
      return;
    }
  }
}

function pickBucketSecs(since: Date, until: Date): number {
  const durationSecs = (until.getTime() - since.getTime()) / 1000;
  return (
    BUCKET_SIZES_SECS.find(
      (bucketSecs) => durationSecs / bucketSecs <= APPROX_INITIAL_BUCKETS,
    ) ?? BUCKET_SIZES_SECS[BUCKET_SIZES_SECS.length - 1]
  );
}

function bucketsToRanges(
  buckets: AppCountLogsResponse_LogBucket[],
  bucketSecs: number,
): LogRange[] {
  return buckets.map((bucket) => {
    const startMs = bucket.bucketStartAt?.getTime() ?? 0;
    return {
      startMs,
      endMs: startMs + bucketSecs * 1000,
      count: bucket.stdoutLogs + bucket.stderrLogs + bucket.systemLogs,
    };
  });
}

function nextSmallerBucketSecs(durationSecs: number): number | null {
  for (const s of BUCKET_SIZES_SECS.slice().reverse()) {
    if (s < durationSecs) {
      return s;
    }
  }
  return null;
}

async function countLogs(
  client: ModalClient,
  filters: LogFilters,
  since: Date,
  until: Date,
  bucketSecs: number,
): Promise<AppCountLogsResponse> {
  const countReq = AppCountLogsRequest.create({
    appId: filters.appId,
    since,
    until,
    source: filters.source,
    functionId: filters.functionId,
    functionCallId: filters.functionCallId,
    searchText: filters.searchText,
    taskId: filters.taskId,
    sandboxId: filters.sandboxId,
    bucketSecs,
  });

  return client.cpClient.appCountLogs(countReq);
}

async function recountRange(
  client: ModalClient,
  filters: LogFilters,
  range: LogRange,
  bucketSecs: number,
): Promise<LogRange[]> {
  const response = await countLogs(
    client,
    filters,
    new Date(range.startMs),
    new Date(range.endMs),
    bucketSecs,
  );
  const subRanges = bucketsToRanges(response.buckets, bucketSecs);

  // A bucket can extend beyond its parent when the bucket size does not evenly
  // divide the parent range.
  const firstRange = subRanges[0];
  if (firstRange !== undefined) {
    firstRange.startMs = Math.max(firstRange.startMs, range.startMs);
  }
  const lastRange = subRanges.at(-1);
  if (lastRange !== undefined) {
    lastRange.endMs = Math.min(lastRange.endMs, range.endMs);
  }

  return subRanges;
}

async function refineDenseRanges(
  client: ModalClient,
  ranges: LogRange[],
  filters: LogFilters,
  maxRanges: number,
  maxIterations: number,
): Promise<LogRange[]> {
  let refined = [...ranges];
  const limit = pLimit(MAX_CONCURRENT_COUNTS);

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const refinements: RangeRefinement[] = refined.flatMap((range, index) => {
      if (range.count <= FETCH_LIMIT) return [];
      const durationSecs = (range.endMs - range.startMs) / 1000;
      const bucketSecs = nextSmallerBucketSecs(durationSecs);
      return bucketSecs === null ? [] : [{ index, range, bucketSecs }];
    });

    if (refinements.length === 0) {
      return refined;
    }

    // Estimate new range count: each refined range of duration D
    // with bucket size S produces ceil(D/S) sub-ranges, replacing 1.
    const estimatedRangeCount = refinements.reduce(
      (total, { range, bucketSecs }) => {
        const durationSecs = (range.endMs - range.startMs) / 1000;
        return total + Math.ceil(durationSecs / bucketSecs) - 1;
      },
      refined.length,
    );
    if (estimatedRangeCount > maxRanges) {
      return refined;
    }

    const replacementEntries = await limit.map(
      refinements,
      async ({ index, range, bucketSecs }) =>
        [
          index,
          await recountRange(client, filters, range, bucketSecs),
        ] as const,
    );

    const replacements = new Map(replacementEntries);
    refined = refined.flatMap(
      (range, index) => replacements.get(index) ?? [range],
    );
  }

  return refined;
}

function buildFetchIntervals(ranges: LogRange[]): FetchInterval[] {
  const intervals: FetchInterval[] = [];
  let currentStartMs: number | undefined;
  let currentEndMs = 0;
  let currentCount = 0;

  for (const { startMs, endMs, count } of ranges) {
    if (count === 0) {
      if (currentStartMs !== undefined) {
        intervals.push({ startMs: currentStartMs, endMs: currentEndMs });
        currentStartMs = undefined;
        currentCount = 0;
      }
      continue;
    }

    if (currentStartMs !== undefined && currentCount + count > FETCH_LIMIT) {
      intervals.push({ startMs: currentStartMs, endMs: currentEndMs });
      currentStartMs = undefined;
      currentCount = 0;
    }

    if (currentStartMs === undefined) {
      currentStartMs = startMs;
      currentCount = count;
    } else {
      currentCount += count;
    }
    currentEndMs = endMs;

    if (currentCount >= INTERVAL_LOG_THRESHOLD) {
      intervals.push({ startMs: currentStartMs, endMs: currentEndMs });
      currentStartMs = undefined;
      currentCount = 0;
    }
  }

  if (currentStartMs !== undefined) {
    intervals.push({ startMs: currentStartMs, endMs: currentEndMs });
  }

  return intervals;
}

async function* fetchLogs(
  client: ModalClient,
  filters: LogFilters,
  since: Date,
  until?: Date,
): AsyncIterable<LogEntry> {
  const effectiveUntil = until ?? new Date();

  if (effectiveUntil.getTime() - since.getTime() > MAX_FETCH_RANGE) {
    throw new Error(
      `Time range cannot exceed ${MAX_FETCH_RANGE / (1000 * 60 * 60 * 24)} days.`,
    );
  }

  const bucketSecs = pickBucketSecs(since, effectiveUntil);

  const countResp = await countLogs(
    client,
    filters,
    since,
    effectiveUntil,
    bucketSecs,
  );

  const ranges = bucketsToRanges(countResp.buckets, bucketSecs);
  const totalLogs = ranges.reduce((total, range) => total + range.count, 0);
  if (totalLogs === 0) {
    return;
  }

  // Trim leading/trailing empty buckets so they don't consume refinement
  // budget. Interior zeros are kept to prevent merging across gaps.

  while (ranges[0]?.count === 0) {
    ranges.shift();
  }
  while (ranges.at(-1)?.count === 0) {
    ranges.pop();
  }

  const refinedRanges = await refineDenseRanges(
    client,
    ranges,
    filters,
    MAX_FETCHES,
    MAX_REFINE_ITERATIONS,
  );

  if (refinedRanges.some((range) => range.count > FETCH_LIMIT)) {
    throw new Error(FETCH_ERROR_MESSAGE);
  }

  const sinceMs = since.getTime();
  const untilMs = effectiveUntil.getTime();
  const intervals = buildFetchIntervals(refinedRanges)
    .map(({ startMs, endMs }) => ({
      startMs: Math.max(startMs, sinceMs),
      endMs: Math.min(endMs, untilMs),
    }))
    .filter(({ startMs, endMs }) => startMs < endMs);

  if (intervals.length > MAX_FETCHES) {
    throw new Error(FETCH_ERROR_MESSAGE);
  }

  const limitFetch = pLimit(MAX_CONCURRENT_FETCHES);

  const fetchResults = intervals.map(({ startMs, endMs }) =>
    limitFetch(() =>
      client.cpClient.appFetchLogs(
        AppFetchLogsRequest.create({
          appId: filters.appId,
          since: new Date(startMs),
          until: new Date(endMs),
          limit: FETCH_LIMIT,
          source: filters.source,
          functionId: filters.functionId,
          functionCallId: filters.functionCallId,
          taskId: filters.taskId,
          sandboxId: filters.sandboxId,
          searchText: filters.searchText,
        }),
      ),
    ).then(
      (response) => ({ status: "fulfilled" as const, response }),
      (error: unknown) => ({ status: "rejected" as const, error }),
    ),
  );

  try {
    for (const resultPromise of fetchResults) {
      const result = await resultPromise;
      if (result.status === "rejected") {
        throw result.error;
      }

      for (const batch of result.response.batches) {
        for (const item of batch.items) {
          yield {
            message: item.data,
            source: fileDescriptorToSource(item.fileDescriptor),
            timestamp: new Date(
              item.timestampNs && item.timestampNs !== "0"
                ? Number(BigInt(item.timestampNs) / 1_000_000n)
                : item.timestamp * 1_000,
            ),
            objectId: filters.objectId,
            contextIds: contextIds(batch, item, filters.objectId),
          };
        }
      }
    }
  } finally {
    limitFetch.clearQueue();
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const finish = () => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    };
    const timeout = setTimeout(finish, ms);
    const onAbort = () => {
      clearTimeout(timeout);
      finish();
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

// Resolves once no log entry has arrived within the configured timeout. The
// promise remains stable across resets so active races keep observing it.
class IdleLogTimer {
  readonly done: Promise<StreamSignal>;

  readonly #timeoutMs: number | undefined;
  readonly #resolve: (signal: StreamSignal) => void;
  #cancelTimer: (() => void) | undefined;

  constructor(timeoutMs: number | undefined) {
    this.#timeoutMs = timeoutMs;
    let resolve!: (signal: StreamSignal) => void;
    this.done = new Promise((promiseResolve) => {
      resolve = promiseResolve;
    });
    this.#resolve = resolve;
    this.reset();
  }

  reset(): void {
    if (this.#timeoutMs === undefined) {
      return;
    }
    this.#clearTimer();
    const timer = setTimeout(() => {
      this.#cancelTimer = undefined;
      this.#resolve({ kind: "idle" });
    }, this.#timeoutMs);
    this.#cancelTimer = () => clearTimeout(timer);
  }

  close(): void {
    this.#clearTimer();
  }

  #clearTimer(): void {
    this.#cancelTimer?.();
    this.#cancelTimer = undefined;
  }
}

// Watches the optional object-specific condition that ends a stream, such as a
// FunctionCall reaching a terminal state.
class LogStreamStopper {
  readonly done: Promise<StreamSignal>;

  readonly #controller = new AbortController();

  constructor(stopStream: (() => Promise<boolean>) | undefined) {
    this.done =
      stopStream === undefined
        ? new Promise(() => {})
        : this.#watch(stopStream);
  }

  close(): void {
    this.#controller.abort();
  }

  async #watch(stopStream: () => Promise<boolean>): Promise<StreamSignal> {
    try {
      while (!this.#controller.signal.aborted) {
        if (await stopStream()) {
          return { kind: "stop" };
        }
        await sleep(STREAM_POLL_INTERVAL_MS, this.#controller.signal);
      }
    } catch (error) {
      return { kind: "stopError", error };
    }

    // The promise intentionally remains pending after cancellation. Consumers
    // stop waiting for it when their stream closes.
    return new Promise(() => {});
  }
}

// Owns the reconnect budget and exponential backoff for interrupted streams.
class LogStreamRetrier {
  #remaining = STREAM_RETRIES;
  #delayMs = 1;

  reset(): void {
    this.#remaining = STREAM_RETRIES;
    this.#delayMs = 1;
  }

  async wait(
    error: unknown,
    stopper: LogStreamStopper,
    idle: IdleLogTimer,
  ): Promise<StreamRetryAction> {
    if (!isRetryableGrpc(error) || this.#remaining <= 0) {
      return { kind: "fail", error };
    }

    this.#remaining -= 1;
    const controller = new AbortController();
    try {
      const action = await Promise.race([
        sleep(this.#delayMs, controller.signal).then(
          () => ({ kind: "reconnect" }) as const,
        ),
        stopper.done,
        idle.done,
      ]);
      if (action.kind === "reconnect") {
        this.#delayMs = Math.min(1_000, this.#delayMs * 10);
      }
      return action;
    } finally {
      controller.abort();
    }
  }
}

function createLogStream(
  params: LogQueryParams,
  lastEntryId: string,
  timeoutSecs: number,
  signal?: AbortSignal,
): AsyncIterable<TaskLogsBatch> {
  return params.client.cpClient.appGetLogs(
    AppGetLogsRequest.create({
      appId: params.appId,
      timeout: timeoutSecs,
      lastEntryId,
      fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_UNSPECIFIED,
      functionId: params.functionId,
      functionCallId: params.functionCallId,
    }),
    { signal },
  );
}

// Owns one AppGetLogs iterator and all of the cancellation details required to
// stop an outstanding next() call without producing an unhandled rejection.
class LogStreamAttempt {
  readonly #controller = new AbortController();
  readonly #iterator: AsyncIterator<TaskLogsBatch>;
  #detached = false;

  constructor(params: LogQueryParams, lastEntryId: string) {
    this.#iterator = createLogStream(
      params,
      lastEntryId,
      STREAM_RPC_TIMEOUT_SECONDS,
      this.#controller.signal,
    )[Symbol.asyncIterator]();
  }

  async next(
    stopper: LogStreamStopper,
    idle: IdleLogTimer,
  ): Promise<StreamAttemptEvent> {
    const nextBatch = this.#iterator.next();
    const event = await Promise.race([
      nextBatch.then(
        (result): StreamAttemptEvent =>
          result.done
            ? { kind: "eof" }
            : { kind: "batch", batch: result.value },
        (error: unknown): StreamAttemptEvent => ({
          kind: "streamError",
          error,
        }),
      ),
      stopper.done,
      idle.done,
    ]);

    if (
      event.kind === "stop" ||
      event.kind === "idle" ||
      event.kind === "stopError"
    ) {
      this.#detached = true;
      this.#controller.abort();
      void nextBatch.catch(() => undefined);
      void this.#iterator.return?.().catch(() => undefined);
    }

    return event;
  }

  async close(): Promise<void> {
    this.#controller.abort();
    if (this.#detached) {
      return;
    }
    try {
      await this.#iterator.return?.();
    } catch {
      // Aborting a nice-grpc stream rejects the iterator during cleanup.
    }
  }
}

function entriesFromBatch(batch: TaskLogsBatch, objectId: string): LogEntry[] {
  return batch.items
    .filter((item) => item.data.length > 0)
    .map((item) => ({
      message: item.data,
      source: fileDescriptorToSource(item.fileDescriptor),
      timestamp: new Date(
        item.timestampNs && item.timestampNs !== "0"
          ? Number(BigInt(item.timestampNs) / 1_000_000n)
          : item.timestamp * 1_000,
      ),
      objectId,
      contextIds: contextIds(batch, item, objectId),
    }));
}

async function* drainLogStream(
  params: LogQueryParams,
  lastEntryId: string,
): AsyncIterable<LogEntry> {
  for await (const batch of createLogStream(
    params,
    lastEntryId,
    STREAM_DRAIN_TIMEOUT_SECONDS,
  )) {
    yield* entriesFromBatch(batch, params.objectId);
    if (batch.appDone) {
      return;
    }
  }
}

async function* streamLogs(
  params: LogQueryParams,
  timeoutMs: number | undefined,
  stopStream?: () => Promise<boolean>,
): AsyncIterable<LogEntry> {
  if (timeoutMs !== undefined && timeoutMs <= 0) {
    return;
  }

  const idle = new IdleLogTimer(timeoutMs);
  const stopper = new LogStreamStopper(stopStream);
  const retrier = new LogStreamRetrier();
  let lastEntryId = "";

  try {
    attempts: while (true) {
      const attempt = new LogStreamAttempt(params, lastEntryId);

      try {
        while (true) {
          const event = await attempt.next(stopper, idle);

          switch (event.kind) {
            case "batch": {
              retrier.reset();
              const batch = event.batch;
              if (batch.entryId) {
                lastEntryId = batch.entryId;
              }
              for (const entry of entriesFromBatch(batch, params.objectId)) {
                idle.reset();
                yield entry;
              }
              if (batch.appDone) {
                return;
              }
              break;
            }

            case "eof":
              continue attempts;

            case "stop":
              yield* drainLogStream(params, lastEntryId);
              return;

            case "idle":
              return;

            case "stopError":
              throw event.error;

            case "streamError": {
              const action = await retrier.wait(event.error, stopper, idle);
              switch (action.kind) {
                case "reconnect":
                  continue attempts;
                case "stop":
                  yield* drainLogStream(params, lastEntryId);
                  return;
                case "idle":
                  return;
                case "stopError":
                case "fail":
                  throw action.error;
              }
            }
          }
        }
      } finally {
        await attempt.close();
      }
    }
  } finally {
    idle.close();
    stopper.close();
  }
}

async function getFunctionCallInfo(
  client: ModalClient,
  functionId: string,
  functionCallId: string,
): Promise<FunctionCallInfo | null> {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const request = FunctionCallGetInfoRequest.create({
        functionId,
        functionCallId,
      });
      const response = await client.cpClient.functionCallGetInfo(request);
      return response.info ?? null;
    } catch (error) {
      if (
        error instanceof ClientError &&
        error.code === Status.NOT_FOUND &&
        attempt < 4
      ) {
        await new Promise((resolve) => setTimeout(resolve, 1_000));
        continue;
      }
      throw error;
    }
  }
  return null;
}

async function determineFunctionCallStop(
  client: ModalClient,
  functionId: string,
  functionCallId: string,
): Promise<boolean> {
  let infoResponse: FunctionCallInfo | null = null;
  try {
    infoResponse = await getFunctionCallInfo(
      client,
      functionId,
      functionCallId,
    );
  } catch (error) {
    if (error instanceof ClientError && error.code === Status.NOT_FOUND) {
      return false;
    }
    if (
      error instanceof ClientError &&
      error.code === Status.RESOURCE_EXHAUSTED
    ) {
      await sleep(1_000);
      return false;
    }
    if (isRetryableGrpc(error)) {
      return false;
    }
    throw error;
  }
  if (!infoResponse) {
    return false;
  }
  const terminalInputs =
    (infoResponse.succeededInputs?.total ?? 0) +
    (infoResponse.failedInputs?.total ?? 0) +
    (infoResponse.timeoutInputs?.total ?? 0) +
    (infoResponse.cancelledInputs?.total ?? 0);
  const totalInputs = infoResponse.totalInputs ?? 0;
  return terminalInputs === totalInputs && totalInputs > 0;
}

class LogsManager {
  constructor(protected readonly params: LogQueryParams) {}

  fetch(params: FunctionLogFetchParams): AsyncIterable<LogEntry> {
    return fetchLogs(
      this.params.client,
      {
        appId: this.params.appId,
        objectId: this.params.objectId,
        functionId: this.params.functionId,
        functionCallId: this.params.functionCallId,
        source: sourceToFileDescriptor(params.source),
        searchText: params.searchText,
      },
      params.since,
      params.until,
    );
  }

  tail(params: LogTailParams = {}): AsyncIterable<LogEntry> {
    const n = params.entries ?? 100;
    return tailLogs(this.params.client, n, {
      appId: this.params.appId,
      objectId: this.params.objectId,
      functionId: this.params.functionId,
      functionCallId: this.params.functionCallId,
      source: sourceToFileDescriptor(params.source),
    });
  }

  stream(params: LogStreamParams = {}): AsyncIterable<LogEntry> {
    return streamLogs(this.params, params.timeoutMs);
  }
}

/** Namespace for Function log APIs. */
export class FunctionLogsManager extends LogsManager implements FunctionLogs {
  /** @ignore */
  constructor(client: ModalClient, objectId: string, appId: string) {
    super({
      client,
      objectId,
      appId,
      functionId: objectId,
      functionCallId: "",
    });
  }

  /**
   * Fetch Function logs corresponding to a UTC time range and optional
   * filters.
   *
   * Entries are returned in chronological order.
   *
   * @returns An async iterable of {@link LogEntry} objects.
   *
   * @example
   * ```typescript
   * import { ModalClient } from "modal";
   *
   * const modal = new ModalClient();
   * const function_ = await modal.functions.fromName("my-app", "train");
   *
   * for await (const entry of function_.logs.fetch({
   *   since: new Date(Date.now() - 4 * 60 * 60 * 1_000),
   *   source: "stdout",
   * })) {
   *   process.stdout.write(entry.message);
   * }
   * ```
   */
  fetch(params: FunctionLogFetchParams): AsyncIterable<LogEntry> {
    return super.fetch(params);
  }

  /**
   * Fetch the most recent Function logs.
   *
   * Entries are returned in chronological order.
   *
   * @returns An async iterable of {@link LogEntry} objects.
   *
   * @example
   * ```typescript
   * import { ModalClient } from "modal";
   *
   * const modal = new ModalClient();
   * const function_ = await modal.functions.fromName("my-app", "train");
   *
   * for await (const entry of function_.logs.tail({ entries: 20 })) {
   *   process.stdout.write(entry.message);
   * }
   * ```
   */
  tail(params: FunctionLogTailParams = {}): AsyncIterable<LogEntry> {
    return super.tail(params);
  }

  /**
   * Stream new Function logs until the timeout is reached.
   *
   * @returns An async iterable of {@link LogEntry} objects as they arrive.
   *
   * @example
   * ```typescript
   * import { ModalClient } from "modal";
   *
   * const modal = new ModalClient();
   * const function_ = await modal.functions.fromName("my-app", "train");
   *
   * for await (const entry of function_.logs.stream({ timeoutMs: 60_000 })) {
   *   process.stdout.write(entry.message);
   * }
   * ```
   */
  stream(params: LogStreamParams = {}): AsyncIterable<LogEntry> {
    return super.stream(params);
  }
}

async function* fetchFunctionCallLogs(
  queryParams: LogQueryParams,
  params: FunctionCallLogFetchParams,
): AsyncIterable<LogEntry> {
  let functionCallStartTime = params.since;
  if (functionCallStartTime === undefined) {
    const info = await getFunctionCallInfo(
      queryParams.client,
      queryParams.functionId,
      queryParams.functionCallId,
    );
    if (info === null) {
      throw new Error(`Function call ${queryParams.objectId} not found.`);
    }
    functionCallStartTime = new Date(info.createdAt * 1_000);
  }

  yield* fetchLogs(
    queryParams.client,
    {
      appId: queryParams.appId,
      objectId: queryParams.objectId,
      functionId: queryParams.functionId,
      functionCallId: queryParams.functionCallId,
      source: sourceToFileDescriptor(params.source),
      searchText: params.searchText,
    },
    functionCallStartTime,
    params.until,
  );
}

/** Namespace for FunctionCall log APIs. */
export class FunctionCallLogsManager
  extends LogsManager
  implements FunctionCallLogs
{
  constructor(
    client: ModalClient,
    objectId: string,
    appId: string,
    functionId: string,
  ) {
    super({ client, objectId, appId, functionId, functionCallId: objectId });
  }

  /**
   * Fetch logs associated with this FunctionCall corresponding to a UTC time
   * range and optional filters.
   *
   * When `since` is omitted, logs are fetched from the start of the
   * FunctionCall. Entries are returned in chronological order.
   *
   * @returns An async iterable of {@link LogEntry} objects.
   *
   * @example
   * ```typescript
   * import { ModalClient } from "modal";
   *
   * const modal = new ModalClient();
   * const function_ = await modal.functions.fromName("my-app", "train");
   * const call = await function_.spawn([]);
   *
   * for await (const entry of call.logs.fetch()) {
   *   console.log(entry.timestamp, entry.message);
   * }
   * ```
   */
  fetch(params: FunctionCallLogFetchParams = {}): AsyncIterable<LogEntry> {
    return fetchFunctionCallLogs(this.params, params);
  }

  /**
   * Fetch the most recent FunctionCall logs.
   *
   * Entries are returned in chronological order.
   *
   * @returns An async iterable of {@link LogEntry} objects.
   *
   * @example
   * ```typescript
   * import { ModalClient } from "modal";
   *
   * const modal = new ModalClient();
   * const function_ = await modal.functions.fromName("my-app", "train");
   * const call = await function_.spawn([]);
   *
   * for await (const entry of call.logs.tail({ entries: 10 })) {
   *   console.log(entry.timestamp, entry.message);
   * }
   * ```
   */
  tail(params: FunctionCallLogTailParams = {}): AsyncIterable<LogEntry> {
    return super.tail(params);
  }

  /**
   * Stream new FunctionCall logs until the timeout is reached or the
   * FunctionCall is observed to have completed.
   *
   * The completion check is best-effort. If completion cannot be determined,
   * the stream continues until the timeout is reached.
   *
   * @returns An async iterable of {@link LogEntry} objects as they arrive.
   *
   * @example
   * ```typescript
   * import { ModalClient } from "modal";
   *
   * const modal = new ModalClient();
   * const function_ = await modal.functions.fromName("my-app", "train");
   * const call = await function_.spawn([]);
   *
   * for await (const entry of call.logs.stream()) {
   *   process.stdout.write(entry.message);
   * }
   * ```
   */
  stream(params: FunctionCallLogStreamParams = {}): AsyncIterable<LogEntry> {
    return streamLogs(this.params, params.timeoutMs, () =>
      determineFunctionCallStop(
        this.params.client,
        this.params.functionId,
        this.params.functionCallId,
      ),
    );
  }
}
