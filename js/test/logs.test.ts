import {
  AppCountLogsRequest,
  AppCountLogsResponse,
  AppCountLogsResponse_LogBucket,
  AppFetchLogsRequest,
  AppFetchLogsResponse,
  AppGetLogsRequest,
  FileDescriptor,
  FunctionCallInfo,
  FunctionCallGetInfoRequest,
  FunctionCallGetInfoResponse,
  FunctionHandleMetadata,
  TaskLogs,
  TaskLogsBatch,
} from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";
import {
  Function_,
  FunctionCall,
  FunctionCallLogsManager,
  FunctionLogsManager,
  type ModalClient,
} from "modal";
import { ClientError, Status } from "nice-grpc";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const now = new Date("2026-07-17T12:00:00Z");

interface TestLog {
  message: string;
  timestamp: Date;
  timestampNs?: string;
  functionCallId?: string;
}

function failedLogStream(error: unknown): AsyncIterable<TaskLogsBatch> {
  return {
    [Symbol.asyncIterator]() {
      return {
        next: () => Promise.reject(error),
      };
    },
  };
}

function createLogsTestContext(logs: TestLog[], expectedRequests: number) {
  const { mockClient, mockCpClient } = createMockModalClients();
  const requests: AppFetchLogsRequest[] = [];

  const handleFetchLogs = (value: unknown) => {
    const request = value as AppFetchLogsRequest;
    requests.push(request);

    const items = logs
      .filter(
        (log) =>
          request.since !== undefined &&
          request.until !== undefined &&
          log.timestamp >= request.since &&
          log.timestamp <= request.until,
      )
      .sort(
        (left, right) => right.timestamp.getTime() - left.timestamp.getTime(),
      )
      .slice(0, request.limit)
      .reverse()
      .map((log) => ({
        data: log.message,
        fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDERR,
        timestamp: log.timestamp.getTime() / 1000,
        timestampNs: log.timestampNs,
        functionCallId: log.functionCallId,
      }));

    return {
      batches:
        items.length > 0
          ? [
              {
                functionId: "fu-123",
                inputId: "in-batch",
                taskId: "ta-batch",
                items,
              },
            ]
          : [],
    };
  };

  for (let i = 0; i < expectedRequests; i++) {
    mockCpClient.handleUnary("/AppFetchLogs", handleFetchLogs);
  }

  const function_ = new Function_(
    mockClient,
    "fu-123",
    undefined,
    FunctionHandleMetadata.create({ appId: "ap-123" }),
  );
  return { logs: function_.logs, mockCpClient, requests };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(now);
});

afterEach(() => {
  vi.useRealTimers();
});

test("tail uses a default request limit of 100 when n is omitted", async () => {
  const sourceLogs = Array.from({ length: 150 }, (_, index) => ({
    message: `line ${index}\n`,
    timestamp: new Date(now.getTime() - index * 1000),
  }));
  const { logs, mockCpClient, requests } = createLogsTestContext(sourceLogs, 1);

  const entries = [];
  for await (const entry of logs.tail({ source: "stderr" })) {
    entries.push(entry);
  }

  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    appId: "ap-123",
    limit: 100,
    source: FileDescriptor.FILE_DESCRIPTOR_STDERR,
    functionId: "fu-123",
  });
  expect(entries).toHaveLength(100);
  expect(entries[0]).toMatchObject({
    message: "line 99\n",
    source: "stderr",
    objectId: "fu-123",
  });
  expect(entries.at(-1)).toMatchObject({
    message: "line 0\n",
    source: "stderr",
    objectId: "fu-123",
  });
  mockCpClient.assertExhausted();
});

test("tail progressively widens its lookback", async () => {
  const sourceLogs = [
    {
      message: "two hours ago\n",
      timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000),
    },
    {
      message: "two days ago\n",
      timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000),
    },
    {
      message: "ten days ago\n",
      timestamp: new Date(now.getTime() - 10 * 24 * 60 * 60 * 1000),
    },
  ];
  const { logs, mockCpClient, requests } = createLogsTestContext(sourceLogs, 4);

  const entries = [];
  for await (const entry of logs.tail({ entries: 3, source: "stderr" })) {
    entries.push(entry);
  }

  expect(
    requests.map((request) => now.getTime() - request.since!.getTime()),
  ).toEqual([
    60 * 60 * 1000,
    24 * 60 * 60 * 1000,
    7 * 24 * 60 * 60 * 1000,
    30 * 24 * 60 * 60 * 1000,
  ]);
  expect(entries.map((entry) => entry.message)).toEqual([
    "ten days ago\n",
    "two days ago\n",
    "two hours ago\n",
  ]);
  mockCpClient.assertExhausted();
});

test("tail populates timestamp and context IDs", async () => {
  const timestamp = new Date(now.getTime() - 1_000);
  const timestampFromNs = new Date(now.getTime() - 2_000);
  const sourceLogs = [
    {
      message: "hello\n",
      timestamp,
      timestampNs: `${timestampFromNs.getTime()}000000`,
      functionCallId: "fc-123",
    },
  ];
  const { logs, mockCpClient } = createLogsTestContext(sourceLogs, 1);

  const entries = [];
  for await (const entry of logs.tail({ entries: 1 })) {
    entries.push(entry);
  }

  expect(entries).toHaveLength(1);
  expect(entries[0]).toMatchObject({
    timestamp: timestampFromNs,
    contextIds: ["fc-123", "in-batch", "ta-batch"],
  });
  mockCpClient.assertExhausted();
});

test("function call fetch defaults since to the call creation time", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const infoRequests: FunctionCallGetInfoRequest[] = [];
  const countRequests: AppCountLogsRequest[] = [];
  const createdAt = new Date("2026-07-17T11:30:00Z");

  mockCpClient.handleUnary("/FunctionCallGetInfo", (value) => {
    infoRequests.push(value as FunctionCallGetInfoRequest);
    return FunctionCallGetInfoResponse.create({
      info: FunctionCallInfo.create({
        createdAt: createdAt.getTime() / 1_000,
      }),
    });
  });
  mockCpClient.handleUnary("/AppCountLogs", (value) => {
    countRequests.push(value as AppCountLogsRequest);
    return AppCountLogsResponse.create({});
  });

  const call = new FunctionCall(mockClient, "fc-123", "ap-123", "fu-123");
  const entries = [];
  for await (const entry of call.logs.fetch({ until: now })) {
    entries.push(entry);
  }

  expect(entries).toEqual([]);
  expect(infoRequests).toEqual([
    expect.objectContaining({
      functionId: "fu-123",
      functionCallId: "fc-123",
    }),
  ]);
  expect(countRequests).toEqual([
    expect.objectContaining({
      appId: "ap-123",
      functionId: "fu-123",
      functionCallId: "fc-123",
      since: createdAt,
      until: now,
    }),
  ]);
  mockCpClient.assertExhausted();
});

test("fetch builds bounded intervals and fetches them in order", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const countRequests: AppCountLogsRequest[] = [];
  const fetchRequests: AppFetchLogsRequest[] = [];
  const since = new Date("2026-07-17T12:00:01Z");
  const until = new Date("2026-07-17T12:00:09Z");
  const bucketStart = new Date("2026-07-17T12:00:00Z");

  mockCpClient.handleUnary("/AppCountLogs", (value) => {
    countRequests.push(value as AppCountLogsRequest);
    return AppCountLogsResponse.create({
      appId: "ap-123",
      buckets: [10, 10, 0, 2_000, 10].map((stdoutLogs, index) =>
        AppCountLogsResponse_LogBucket.create({
          bucketStartAt: new Date(bucketStart.getTime() + index * 2_000),
          stdoutLogs,
        }),
      ),
    });
  });

  for (let index = 0; index < 3; index++) {
    mockCpClient.handleUnary("/AppFetchLogs", (value) => {
      const request = value as AppFetchLogsRequest;
      fetchRequests.push(request);
      return AppFetchLogsResponse.create({
        batches: [
          TaskLogsBatch.create({
            items: [
              TaskLogs.create({
                data: `interval ${index}\n`,
                fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDERR,
                timestamp: request.since!.getTime() / 1_000,
              }),
            ],
          }),
        ],
      });
    });
  }

  const function_ = new Function_(
    mockClient,
    "fu-123",
    undefined,
    FunctionHandleMetadata.create({ appId: "ap-123" }),
  );
  const entries = [];
  for await (const entry of function_.logs.fetch({
    since,
    until,
    source: "stderr",
    searchText: "needle",
  })) {
    entries.push(entry);
  }

  expect(countRequests).toHaveLength(1);
  expect(countRequests[0]).toMatchObject({
    appId: "ap-123",
    since,
    until,
    bucketSecs: 2,
    source: FileDescriptor.FILE_DESCRIPTOR_STDERR,
    functionId: "fu-123",
    searchText: "needle",
  });
  expect(
    fetchRequests.map((request) => [request.since, request.until]),
  ).toEqual([
    [since, new Date("2026-07-17T12:00:04Z")],
    [new Date("2026-07-17T12:00:06Z"), new Date("2026-07-17T12:00:08Z")],
    [new Date("2026-07-17T12:00:08Z"), until],
  ]);
  for (const request of fetchRequests) {
    expect(request).toMatchObject({
      appId: "ap-123",
      limit: 20_000,
      source: FileDescriptor.FILE_DESCRIPTOR_STDERR,
      functionId: "fu-123",
      searchText: "needle",
    });
  }
  expect(entries.map((entry) => entry.message)).toEqual([
    "interval 0\n",
    "interval 1\n",
    "interval 2\n",
  ]);
  mockCpClient.assertExhausted();
});

test("fetch refines all dense ranges in the same iteration", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const countRequests: AppCountLogsRequest[] = [];
  const fetchRequests: AppFetchLogsRequest[] = [];
  const since = new Date("2026-07-17T12:00:00Z");
  const until = new Date("2026-07-17T13:00:00Z");

  mockCpClient.handleUnary("/AppCountLogs", (value) => {
    countRequests.push(value as AppCountLogsRequest);
    return AppCountLogsResponse.create({
      buckets: [0, 60].map((offsetSecs) =>
        AppCountLogsResponse_LogBucket.create({
          bucketStartAt: new Date(since.getTime() + offsetSecs * 1_000),
          stdoutLogs: 20_001,
        }),
      ),
    });
  });

  for (const parentOffsetSecs of [0, 60]) {
    mockCpClient.handleUnary("/AppCountLogs", (value) => {
      countRequests.push(value as AppCountLogsRequest);
      return AppCountLogsResponse.create({
        buckets: [0, 30].map((offsetSecs) =>
          AppCountLogsResponse_LogBucket.create({
            bucketStartAt: new Date(
              since.getTime() + (parentOffsetSecs + offsetSecs) * 1_000,
            ),
            stdoutLogs: 100,
          }),
        ),
      });
    });
  }

  mockCpClient.handleUnary("/AppFetchLogs", (value) => {
    fetchRequests.push(value as AppFetchLogsRequest);
    return AppFetchLogsResponse.create({});
  });

  const function_ = new Function_(
    mockClient,
    "fu-123",
    undefined,
    FunctionHandleMetadata.create({ appId: "ap-123" }),
  );
  for await (const _ of function_.logs.fetch({ since, until })) {
    // The mock fetch response contains no entries.
  }

  expect(countRequests).toHaveLength(3);
  expect(countRequests[0].bucketSecs).toBe(60);
  expect(
    countRequests.slice(1).map((request) => ({
      since: request.since,
      until: request.until,
      bucketSecs: request.bucketSecs,
    })),
  ).toEqual([
    {
      since,
      until: new Date("2026-07-17T12:01:00Z"),
      bucketSecs: 30,
    },
    {
      since: new Date("2026-07-17T12:01:00Z"),
      until: new Date("2026-07-17T12:02:00Z"),
      bucketSecs: 30,
    },
  ]);
  expect(fetchRequests).toHaveLength(1);
  expect(fetchRequests[0]).toMatchObject({
    since,
    until: new Date("2026-07-17T12:02:00Z"),
    limit: 20_000,
  });
  mockCpClient.assertExhausted();
});

test("stream resumes from the last entry after a transient error", async () => {
  const requests: AppGetLogsRequest[] = [];
  let attempt = 0;
  const cpClient = {
    appGetLogs(request: AppGetLogsRequest) {
      requests.push(request);
      attempt += 1;
      if (attempt === 1) {
        return (async function* () {
          yield TaskLogsBatch.create({
            entryId: "1-0",
            items: [
              TaskLogs.create({
                data: "before reconnect\n",
                fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
              }),
            ],
          });
          throw new ClientError(
            "/modal.client.ModalClient/AppGetLogs",
            Status.UNAVAILABLE,
            "transient",
          );
        })();
      }
      return (async function* () {
        yield TaskLogsBatch.create({
          entryId: "2-0",
          appDone: true,
          items: [
            TaskLogs.create({
              data: "after reconnect\n",
              fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDERR,
            }),
          ],
        });
      })();
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );

  const entriesPromise = (async () => {
    const entries = [];
    for await (const entry of logs.stream()) {
      entries.push(entry);
    }
    return entries;
  })();
  await vi.advanceTimersByTimeAsync(1);
  const entries = await entriesPromise;

  expect(entries.map((entry) => entry.message)).toEqual([
    "before reconnect\n",
    "after reconnect\n",
  ]);
  expect(requests).toHaveLength(2);
  expect(requests[0]).toMatchObject({
    appId: "ap-123",
    functionId: "fu-123",
    lastEntryId: "",
    timeout: 55,
  });
  expect(requests[1].lastEntryId).toBe("1-0");
});

test("stream stops after exhausting its retry budget", async () => {
  let attempts = 0;
  const streamError = new ClientError(
    "/modal.client.ModalClient/AppGetLogs",
    Status.UNAVAILABLE,
    "transient",
  );
  const cpClient = {
    appGetLogs() {
      attempts += 1;
      return failedLogStream(streamError);
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );

  const resultPromise = (async () => {
    try {
      for await (const _entry of logs.stream()) {
        // The stream fails before yielding any entries.
      }
    } catch (error) {
      return error;
    }
  })();

  await vi.runAllTimersAsync();

  expect(await resultPromise).toBe(streamError);
  expect(attempts).toBe(11);
});

test("stream resets its retry budget after receiving a batch", async () => {
  let attempts = 0;
  const cpClient = {
    appGetLogs() {
      attempts += 1;
      const attempt = attempts;
      return (async function* () {
        if (attempt <= 10 || (attempt >= 12 && attempt <= 21)) {
          throw new ClientError(
            "/modal.client.ModalClient/AppGetLogs",
            Status.UNAVAILABLE,
            "transient",
          );
        }
        yield TaskLogsBatch.create({
          entryId: attempt === 11 ? "1-0" : "2-0",
          appDone: attempt === 22,
          items: [
            TaskLogs.create({
              data: attempt === 11 ? "healthy\n" : "done\n",
              fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
            }),
          ],
        });
      })();
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );

  const entriesPromise = (async () => {
    const entries = [];
    for await (const entry of logs.stream()) {
      entries.push(entry);
    }
    return entries;
  })();

  await vi.runAllTimersAsync();

  expect((await entriesPromise).map((entry) => entry.message)).toEqual([
    "healthy\n",
    "done\n",
  ]);
  expect(attempts).toBe(22);
});

test("stream idle timeout interrupts retry backoff", async () => {
  let attempts = 0;
  const cpClient = {
    appGetLogs() {
      attempts += 1;
      return failedLogStream(
        new ClientError(
          "/modal.client.ModalClient/AppGetLogs",
          Status.UNAVAILABLE,
          "transient",
        ),
      );
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );

  const entriesPromise = (async () => {
    const entries = [];
    for await (const entry of logs.stream({ timeoutMs: 50 })) {
      entries.push(entry);
    }
    return entries;
  })();

  await vi.advanceTimersByTimeAsync(50);

  expect(await entriesPromise).toEqual([]);
  expect(attempts).toBe(3);
});

test("function call completion interrupts retry backoff", async () => {
  let statusChecks = 0;
  let streamAttempts = 0;
  let drainAttempts = 0;
  const cpClient = {
    async functionCallGetInfo() {
      statusChecks += 1;
      return {
        info:
          statusChecks === 1
            ? { totalInputs: 1 }
            : { totalInputs: 1, succeededInputs: { total: 1 } },
      };
    },
    appGetLogs(request: AppGetLogsRequest) {
      if (request.timeout === 0.5) {
        drainAttempts += 1;
        return (async function* () {
          yield TaskLogsBatch.create({ appDone: true });
        })();
      }
      streamAttempts += 1;
      return failedLogStream(
        new ClientError(
          "/modal.client.ModalClient/AppGetLogs",
          Status.UNAVAILABLE,
          "transient",
        ),
      );
    },
  };
  const logs = new FunctionCallLogsManager(
    { cpClient } as unknown as ModalClient,
    "fc-123",
    "ap-123",
    "fu-123",
  );

  const entriesPromise = (async () => {
    const entries = [];
    for await (const entry of logs.stream()) {
      entries.push(entry);
    }
    return entries;
  })();

  await vi.advanceTimersByTimeAsync(1_000);

  expect(await entriesPromise).toEqual([]);
  expect(statusChecks).toBe(2);
  expect(streamAttempts).toBe(4);
  expect(drainAttempts).toBe(1);
});

test("stream timeout resets after each log entry", async () => {
  const cpClient = {
    appGetLogs(_request: AppGetLogsRequest, options: { signal?: AbortSignal }) {
      return (async function* () {
        await new Promise((resolve) => setTimeout(resolve, 40));
        yield TaskLogsBatch.create({
          entryId: "1-0",
          items: [
            TaskLogs.create({
              data: "first\n",
              fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
            }),
          ],
        });
        await new Promise((resolve) => setTimeout(resolve, 40));
        yield TaskLogsBatch.create({
          entryId: "2-0",
          items: [
            TaskLogs.create({
              data: "second\n",
              fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
            }),
          ],
        });
        await new Promise<void>((resolve) => {
          options.signal?.addEventListener("abort", () => resolve(), {
            once: true,
          });
        });
      })();
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );
  const iterator = logs.stream({ timeoutMs: 50 })[Symbol.asyncIterator]();

  const firstPromise = iterator.next();
  await vi.advanceTimersByTimeAsync(40);
  expect((await firstPromise).value?.message).toBe("first\n");

  const secondPromise = iterator.next();
  await vi.advanceTimersByTimeAsync(40);
  expect((await secondPromise).value?.message).toBe("second\n");

  const donePromise = iterator.next();
  await vi.advanceTimersByTimeAsync(50);
  expect(await donePromise).toEqual({ done: true, value: undefined });
});

test("stream timeout does not reset after an empty batch", async () => {
  const cpClient = {
    appGetLogs(_request: AppGetLogsRequest, options: { signal?: AbortSignal }) {
      return (async function* () {
        await new Promise((resolve) => setTimeout(resolve, 30));
        yield TaskLogsBatch.create({ entryId: "1-0" });

        const reachedLateBatch = await new Promise<boolean>((resolve) => {
          const timeout = setTimeout(() => resolve(true), 40);
          options.signal?.addEventListener(
            "abort",
            () => {
              clearTimeout(timeout);
              resolve(false);
            },
            { once: true },
          );
        });
        if (reachedLateBatch) {
          yield TaskLogsBatch.create({
            items: [
              TaskLogs.create({
                data: "late\n",
                fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
              }),
            ],
          });
        }
      })();
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );
  const iterator = logs.stream({ timeoutMs: 50 })[Symbol.asyncIterator]();

  const resultPromise = iterator.next();
  await vi.advanceTimersByTimeAsync(50);

  expect(await resultPromise).toEqual({ done: true, value: undefined });
});

test("breaking out of stream cancels the active RPC", async () => {
  let streamClosed = false;
  const cpClient = {
    appGetLogs() {
      return (async function* () {
        try {
          yield TaskLogsBatch.create({
            items: [
              TaskLogs.create({
                data: "one line\n",
                fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
              }),
            ],
          });
          await new Promise(() => {});
        } finally {
          streamClosed = true;
        }
      })();
    },
  };
  const logs = new FunctionLogsManager(
    { cpClient } as unknown as ModalClient,
    "fu-123",
    "ap-123",
  );

  for await (const _entry of logs.stream()) {
    break;
  }

  expect(streamClosed).toBe(true);
});

test("function call stream drains trailing logs after completion", async () => {
  const requests: AppGetLogsRequest[] = [];
  const cpClient = {
    async functionCallGetInfo() {
      return {
        info: {
          totalInputs: 2,
          succeededInputs: { total: 1 },
          failedInputs: { total: 1 },
        },
      };
    },
    appGetLogs(request: AppGetLogsRequest, options?: { signal?: AbortSignal }) {
      requests.push(request);
      if (request.timeout === 0.5) {
        return (async function* () {
          yield TaskLogsBatch.create({
            appDone: true,
            items: [
              TaskLogs.create({
                data: "trailing\n",
                fileDescriptor: FileDescriptor.FILE_DESCRIPTOR_STDOUT,
              }),
            ],
          });
        })();
      }
      return (async function* () {
        await new Promise<void>((resolve) => {
          options?.signal?.addEventListener("abort", () => resolve(), {
            once: true,
          });
        });
        if (!options?.signal?.aborted) {
          yield TaskLogsBatch.create({});
        }
      })();
    },
  };
  const logs = new FunctionCallLogsManager(
    { cpClient } as unknown as ModalClient,
    "fc-123",
    "ap-123",
    "fu-123",
  );

  const entries = [];
  for await (const entry of logs.stream()) {
    entries.push(entry);
  }

  expect(entries.map((entry) => entry.message)).toEqual(["trailing\n"]);
  expect(requests.map((request) => request.timeout)).toEqual([55, 0.5]);
  expect(requests[1].lastEntryId).toBe("");
});
