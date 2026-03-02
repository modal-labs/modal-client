import { v4 as uuidv4 } from "uuid";
import {
  CallOptions,
  Client,
  ClientError,
  ClientMiddleware,
  ClientMiddlewareCall,
  createChannel,
  createClientFactory,
  Metadata,
  Status,
} from "nice-grpc";
import { AppService } from "./app";
import { CloudBucketMountService } from "./cloud_bucket_mount";
import { ClsService } from "./cls";
import { FunctionService } from "./function";
import { FunctionCallService } from "./function_call";
import { ImageService } from "./image";
import { ProxyService } from "./proxy";
import { QueueService } from "./queue";
import { SandboxService } from "./sandbox";
import { SecretService } from "./secret";
import { VolumeService } from "./volume";

import { ClientType, ModalClientDefinition } from "../proto/modal_proto/api";
import { getProfile, type Profile } from "./config";
import { AuthTokenManager } from "./auth_token_manager";
import { getSDKVersion } from "./version";
import { checkForRenamedParams } from "./validation";
import { createLogger, type Logger, type LogLevel } from "./logger";

export interface ModalClientParams {
  tokenId?: string;
  tokenSecret?: string;
  environment?: string;
  endpoint?: string;
  timeoutMs?: number;
  maxRetries?: number;
  logger?: Logger;
  logLevel?: LogLevel;
  /**
   * Custom gRPC middleware to be applied to all API calls.
   * These middleware are appended after Modal's built-in middleware
   * (authentication, retry logic, and timeouts), allowing you to add
   * telemetry, tracing, or other observability features.
   *
   * Note that the Modal gRPC API is not considered a public API, and
   * can change without warning.
   */
  grpcMiddleware?: ClientMiddleware[];
  /** @ignore */
  cpClient?: ModalGrpcClient;
}

export type ModalGrpcClient = Client<
  typeof ModalClientDefinition,
  TimeoutOptions & RetryOptions
>;

/**
 * The main client for interacting with Modal's cloud infrastructure.
 *
 * ModalClient provides access to all Modal services through service properties.
 * Create a client instance and use its service properties to manage {@link App}s,
 * {@link Function_ Function}s, * {@link Sandbox}es, and other Modal resources.
 *
 * @example
 * ```typescript
 * import { ModalClient } from "modal";
 *
 * const modal = new ModalClient();
 *
 * const app = await modal.apps.fromName("my-app");
 * const image = modal.images.fromRegistry("python:3.13");
 * const sb = await modal.sandboxes.create(app, image);
 * ```
 */
export class ModalClient {
  readonly apps: AppService;
  readonly cloudBucketMounts: CloudBucketMountService;
  readonly cls: ClsService;
  readonly functions: FunctionService;
  readonly functionCalls: FunctionCallService;
  readonly images: ImageService;
  readonly proxies: ProxyService;
  readonly queues: QueueService;
  readonly sandboxes: SandboxService;
  readonly secrets: SecretService;
  readonly volumes: VolumeService;

  /** @ignore */
  readonly cpClient: ModalGrpcClient;
  readonly profile: Profile;
  readonly logger: Logger;

  private ipClients: Map<string, ModalGrpcClient>;
  private authTokenManager: AuthTokenManager | null = null;
  private customMiddleware: ClientMiddleware[];

  constructor(params?: ModalClientParams) {
    checkForRenamedParams(params, { timeout: "timeoutMs" });

    const baseProfile = getProfile(process.env["MODAL_PROFILE"]);
    this.profile = {
      ...baseProfile,
      ...(params?.tokenId && { tokenId: params.tokenId }),
      ...(params?.tokenSecret && { tokenSecret: params.tokenSecret }),
      ...(params?.environment && { environment: params.environment }),
    };

    const logLevelValue = params?.logLevel || this.profile.logLevel || "";
    this.logger = createLogger(params?.logger, logLevelValue);
    this.logger.debug(
      "Initializing Modal client",
      "version",
      getSDKVersion(),
      "server_url",
      this.profile.serverUrl,
    );

    this.customMiddleware = params?.grpcMiddleware ?? [];
    this.ipClients = new Map();
    this.cpClient = params?.cpClient ?? this.createClient(this.profile);

    this.logger.debug("Modal client initialized successfully");

    this.apps = new AppService(this);
    this.cloudBucketMounts = new CloudBucketMountService(this);
    this.cls = new ClsService(this);
    this.functions = new FunctionService(this);
    this.functionCalls = new FunctionCallService(this);
    this.images = new ImageService(this);
    this.proxies = new ProxyService(this);
    this.queues = new QueueService(this);
    this.sandboxes = new SandboxService(this);
    this.secrets = new SecretService(this);
    this.volumes = new VolumeService(this);
  }

  environmentName(environment?: string): string {
    return environment || this.profile.environment || "";
  }

  imageBuilderVersion(version?: string): string {
    return version || this.profile.imageBuilderVersion || "2024.10";
  }

  /** @ignore */
  ipClient(serverUrl: string): ModalGrpcClient {
    const existing = this.ipClients.get(serverUrl);
    if (existing) {
      return existing;
    }

    this.logger.debug("Creating input plane client", "server_url", serverUrl);
    const profile = { ...this.profile, serverUrl };
    const newClient = this.createClient(profile);
    this.ipClients.set(serverUrl, newClient);
    return newClient;
  }

  close(): void {
    this.logger.debug("Closing Modal client");
    this.authTokenManager = null;
    this.logger.debug("Modal client closed");
  }

  version(): string {
    return getSDKVersion();
  }

  private createClient(profile: Profile): ModalGrpcClient {
    // Channels don't do anything until you send a request on them.
    // Ref: https://github.com/modal-labs/modal-client/blob/main/modal/_utils/grpc_utils.py
    const channel = createChannel(profile.serverUrl, undefined, {
      "grpc.max_receive_message_length": 100 * 1024 * 1024,
      "grpc.max_send_message_length": 100 * 1024 * 1024,
      "grpc-node.flow_control_window": 64 * 1024 * 1024,
      "grpc.keepalive_time_ms": 30000,
      "grpc.keepalive_timeout_ms": 10000,
      "grpc.keepalive_permit_without_calls": 1,
    });
    let factory = createClientFactory()
      .use(this.authMiddleware(profile))
      .use(this.retryMiddleware())
      .use(timeoutMiddleware);

    for (const middleware of this.customMiddleware) {
      factory = factory.use(middleware);
    }

    return factory.create(ModalClientDefinition, channel);
  }

  /** Middleware to retry transient errors and timeouts for unary requests. */
  private retryMiddleware(): ClientMiddleware<RetryOptions> {
    const logger = this.logger;
    return async function* retryMiddleware<Request, Response>(
      call: ClientMiddlewareCall<Request, Response>,
      options: CallOptions & RetryOptions,
    ) {
      const {
        retries = 3,
        baseDelay = 100,
        maxDelay = 1000,
        delayFactor = 2,
        additionalStatusCodes = [],
        signal,
        ...restOptions
      } = options;

      if (call.requestStream || call.responseStream || !retries) {
        // Don't retry streaming calls, or if retries are disabled.
        return yield* call.next(call.request, restOptions);
      }

      const retryableCodes = new Set([
        ...retryableGrpcStatusCodes,
        ...additionalStatusCodes,
      ]);

      // One idempotency key for the whole call (all attempts).
      const idempotencyKey = uuidv4();

      const startTime = Date.now();
      let attempt = 0;
      let delayMs = baseDelay;

      logger.debug("Sending gRPC request", "method", call.method.path);

      while (true) {
        // Clone/augment metadata for this attempt.
        const metadata = new Metadata(restOptions.metadata ?? {});

        metadata.set("x-idempotency-key", idempotencyKey);
        metadata.set("x-retry-attempt", String(attempt));
        if (attempt > 0) {
          metadata.set(
            "x-retry-delay",
            ((Date.now() - startTime) / 1000).toFixed(3),
          );
        }

        try {
          // Forward the call.
          return yield* call.next(call.request, {
            ...restOptions,
            metadata,
            signal,
          });
        } catch (err) {
          // Immediately propagate non-retryable situations.
          if (
            !(err instanceof ClientError) ||
            !retryableCodes.has(err.code) ||
            attempt >= retries
          ) {
            if (attempt === retries && attempt > 0) {
              logger.debug(
                "Final retry attempt failed",
                "error",
                err,
                "retries",
                attempt,
                "delay",
                delayMs,
                "method",
                call.method.path,
                "idempotency_key",
                idempotencyKey.substring(0, 8),
              );
            }
            throw err;
          }

          if (attempt > 0) {
            logger.debug(
              "Retryable failure",
              "error",
              err,
              "retries",
              attempt,
              "delay",
              delayMs,
              "method",
              call.method.path,
              "idempotency_key",
              idempotencyKey.substring(0, 8),
            );
          }

          // Exponential back-off with a hard cap.
          await sleep(delayMs, signal);
          delayMs = Math.min(delayMs * delayFactor, maxDelay);
          attempt += 1;
        }
      }
    };
  }

  private authMiddleware(profile: Profile): ClientMiddleware {
    const getOrCreateAuthTokenManager = () => {
      if (!this.authTokenManager) {
        this.authTokenManager = new AuthTokenManager(
          this.cpClient,
          this.logger,
        );
      }
      return this.authTokenManager;
    };

    return async function* authMiddleware<Request, Response>(
      call: ClientMiddlewareCall<Request, Response>,
      options: CallOptions,
    ) {
      if (!profile.tokenId || !profile.tokenSecret) {
        throw new Error(
          `Profile is missing token_id or token_secret. Please set them in .modal.toml, or as environment variables, or via ModalClient constructor.`,
        );
      }
      const { tokenId, tokenSecret } = profile;

      options.metadata ??= new Metadata();
      options.metadata.set(
        "x-modal-client-type",
        String(ClientType.CLIENT_TYPE_LIBMODAL_JS),
      );
      options.metadata.set("x-modal-client-version", "1.0.0"); // CLIENT VERSION: Behaves like this Python SDK version
      options.metadata.set(
        "x-modal-libmodal-version",
        `modal-js/${getSDKVersion()}`,
      );
      options.metadata.set("x-modal-token-id", tokenId);
      options.metadata.set("x-modal-token-secret", tokenSecret);

      // Skip auth token for AuthTokenGet requests to prevent it from getting stuck
      if (call.method.path !== "/modal.client.ModalClient/AuthTokenGet") {
        const tokenManager = getOrCreateAuthTokenManager();
        // getToken() will automatically wait if initial fetch is in progress
        const token = await tokenManager.getToken();
        if (token) {
          options.metadata.set("x-modal-auth-token", token);
        }
      }

      return yield* call.next(call.request, options);
    };
  }
}

export type TimeoutOptions = {
  /** Timeout for this call, interpreted as a duration in milliseconds */
  timeoutMs?: number;
};

/** gRPC client middleware to set timeout and retries on a call. */
export const timeoutMiddleware: ClientMiddleware<TimeoutOptions> =
  async function* timeoutMiddleware(call, options) {
    if (!options.timeoutMs || options.signal?.aborted) {
      return yield* call.next(call.request, options);
    }

    const { timeoutMs, signal: origSignal, ...restOptions } = options;
    const abortController = new AbortController();
    const abortListener = () => abortController.abort();
    origSignal?.addEventListener("abort", abortListener);

    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      abortController.abort();
    }, timeoutMs);

    try {
      return yield* call.next(call.request, {
        ...restOptions,
        signal: abortController.signal,
      });
    } finally {
      origSignal?.removeEventListener("abort", abortListener);
      clearTimeout(timer);

      if (timedOut) {
        // eslint-disable-next-line no-unsafe-finally
        throw new ClientError(
          call.method.path,
          Status.DEADLINE_EXCEEDED,
          `Timed out after ${timeoutMs}ms`,
        );
      }
    }
  };

const retryableGrpcStatusCodes = new Set([
  Status.DEADLINE_EXCEEDED,
  Status.UNAVAILABLE,
  Status.CANCELLED,
  Status.INTERNAL,
  Status.UNKNOWN,
]);

export function isRetryableGrpc(err: unknown) {
  if (err instanceof ClientError) {
    return retryableGrpcStatusCodes.has(err.code);
  }
  return false;
}

/** Sleep helper that can be cancelled via an AbortSignal. */
const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(signal.reason);
      },
      { once: true },
    );
  });

type RetryOptions = {
  /** Number of retries to take. */
  retries?: number;

  /** Base delay in milliseconds. */
  baseDelay?: number;

  /** Maximum delay in milliseconds. */
  maxDelay?: number;

  /** Exponential factor to multiply successive delays. */
  delayFactor?: number;

  /** Additional status codes to retry. */
  additionalStatusCodes?: Status[];
};

// Legacy default client - lazily initialized
let defaultClient: ModalClient | undefined;

// Initialization options for the default client (from initializeClient)
let defaultClientOptions: ModalClientParams | undefined;

export function getDefaultClient(): ModalClient {
  if (!defaultClient) {
    defaultClient = new ModalClient(defaultClientOptions);
  }
  return defaultClient;
}

// Legacy client export for backward compatibility - proxies to control plane client
export const client = new Proxy({} as ModalGrpcClient, {
  get(_target, prop) {
    return getDefaultClient().cpClient[prop as keyof ModalGrpcClient];
  },
});

/**
 * @deprecated Use {@link ModalClient `new ModalClient()`} instead.
 */
export type ClientOptions = {
  tokenId: string;
  tokenSecret: string;
  environment?: string;
};

/**
 * @deprecated Use {@link ModalClient `new ModalClient()`} instead.
 */
// eslint-disable-next-line @typescript-eslint/no-deprecated
export function initializeClient(options: ClientOptions) {
  defaultClientOptions = {
    tokenId: options.tokenId,
    tokenSecret: options.tokenSecret,
    environment: options.environment,
  };
  defaultClient = new ModalClient(defaultClientOptions);
}

/**
 * Stops the auth token refresh.
 * @deprecated Use {@link ModalClient#close modalClient.close()} instead.
 */
export function close() {
  if (defaultClient) {
    defaultClient.close();
  }
}
