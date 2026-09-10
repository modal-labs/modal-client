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
import { RPCRetryPolicy, RPCStatus } from "../proto/modal_proto/api";
import { AppService } from "./app";
import { CloudBucketMountService } from "./cloud_bucket_mount";
import { ClsService } from "./cls";
import { FunctionService } from "./function";
import { FunctionCallService } from "./function_call";
import { ImageService } from "./image";
import { ProxyService } from "./proxy";
import { QueueService } from "./queue";
import { SandboxService } from "./sandbox";
import { SandboxSnapshotService } from "./sandbox_snapshot";
import { SecretService } from "./secret";
import { VolumeService } from "./volume";

import { ClientType, ModalClientDefinition } from "../proto/modal_proto/api";
import { getProfile, type Profile } from "./config";
import { AuthTokenManager } from "./auth_token_manager";
import { getSDKVersion } from "./version";
import { checkForRenamedParams } from "./validation";
import { createLogger, type Logger, type LogLevel } from "./logger";
import { EnvironmentManager } from "./environment";
import { InvalidError } from "./errors";
import { mintOAuthClientAssertion, parseOAuthJwtKey } from "./oauth";
import type { KeyObject } from "node:crypto";

export interface ModalClientParams {
  tokenId?: string;
  tokenSecret?: string;
  /** OAuth refresh token returned by Modal's token endpoint. */
  oauthRefreshToken?: string;
  /** Modal-issued OAuth client ID, with an `oc-` prefix. */
  oauthClientId?: string;
  /** Modal-issued OAuth client secret, with an `ov-` prefix. */
  oauthClientSecret?: string;
  /** Unencrypted RSA private key encoded as PEM, with literal or escaped newlines. */
  oauthJwtKey?: string;
  environment?: string;
  endpoint?: string;
  timeoutMs?: number;
  maxRetries?: number;
  maxThrottleWaitSecs?: number;
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
 * {@link Function_ Function}s, {@link Sandbox}es, and other Modal resources.
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
  readonly sandboxSnapshots: SandboxSnapshotService;
  readonly secrets: SecretService;
  readonly volumes: VolumeService;

  /** @ignore */
  readonly cpClient: ModalGrpcClient;
  readonly profile: Profile;
  readonly logger: Logger;

  private ipClients: Map<string, ModalGrpcClient>;
  private authTokenManager: AuthTokenManager | null = null;
  private customMiddleware: ClientMiddleware[];
  private environmentManager: EnvironmentManager;
  private oauthJwtKey?: KeyObject;

  constructor(params?: ModalClientParams) {
    checkForRenamedParams(params, { timeout: "timeoutMs" });

    const baseProfile = getProfile(process.env["MODAL_PROFILE"]);
    const hasTokenParams =
      params?.tokenId !== undefined || params?.tokenSecret !== undefined;
    const hasOAuthParams =
      params?.oauthRefreshToken !== undefined ||
      params?.oauthClientId !== undefined ||
      params?.oauthClientSecret !== undefined ||
      params?.oauthJwtKey !== undefined;
    this.profile = {
      ...baseProfile,
      ...(params?.environment && { environment: params.environment }),
      ...(params?.maxThrottleWaitSecs !== undefined && {
        maxThrottleWaitSecs: params.maxThrottleWaitSecs,
      }),
    };
    if (hasTokenParams) {
      if (params?.tokenId !== undefined) {
        this.profile.tokenId = params.tokenId;
      }
      if (params?.tokenSecret !== undefined) {
        this.profile.tokenSecret = params.tokenSecret;
      }
      this.profile.oauthRefreshToken = undefined;
      this.profile.oauthClientId = undefined;
      this.profile.oauthClientSecret = undefined;
      this.profile.oauthJwtKey = undefined;
    } else if (hasOAuthParams) {
      this.profile.tokenId = undefined;
      this.profile.tokenSecret = undefined;
      this.profile.oauthRefreshToken = params?.oauthRefreshToken;
      this.profile.oauthClientId = params?.oauthClientId;
      this.profile.oauthClientSecret = params?.oauthClientSecret;
      this.profile.oauthJwtKey = params?.oauthJwtKey;
    }
    this.validateProfileCredentials(hasOAuthParams);

    this.oauthJwtKey = this.profile.oauthJwtKey
      ? parseOAuthJwtKey(this.profile.oauthJwtKey)
      : undefined;

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
    this.sandboxSnapshots = new SandboxSnapshotService(this);
    this.secrets = new SecretService(this);
    this.volumes = new VolumeService(this);
    this.environmentManager = new EnvironmentManager(
      this.cpClient,
      this.logger,
    );
  }

  environmentName(environment?: string): string {
    return environment || this.profile.environment || "";
  }

  private validateProfileCredentials(
    hasExplicitOAuthCredentials: boolean,
  ): void {
    const hasTokenCredentials = Boolean(
      this.profile.tokenId || this.profile.tokenSecret,
    );
    const hasOAuthCredentials =
      hasExplicitOAuthCredentials ||
      Boolean(
        this.profile.oauthRefreshToken ||
          this.profile.oauthClientId ||
          this.profile.oauthClientSecret ||
          this.profile.oauthJwtKey,
      );
    if (hasTokenCredentials && hasOAuthCredentials) {
      throw new InvalidError(
        "Modal token credentials and OAuth credentials cannot both be configured.",
      );
    }
    if (
      hasOAuthCredentials &&
      (!this.profile.oauthRefreshToken ||
        !this.profile.oauthClientId ||
        Boolean(this.profile.oauthClientSecret) ===
          Boolean(this.profile.oauthJwtKey))
    ) {
      throw new InvalidError(
        "OAuth refresh token, client ID, and exactly one of client secret or JWT key must all be configured.",
      );
    }
  }

  /**
   * Returns the image builder version by querying the server where the local profile takes
   * precedence.
   *
   * The image builder version is an environment-scoped server setting, so pass the environment
   * the image will be built in (e.g. an App's environment) to fetch the correct version. When
   * omitted, the profile's default environment is used.
   */
  async getImageBuilderVersion(environmentName?: string): Promise<string> {
    if (
      this.profile.imageBuilderVersion != null &&
      this.profile.imageBuilderVersion !== ""
    ) {
      return this.profile.imageBuilderVersion;
    }
    return this.environmentManager.getImageBuilderVersion(
      environmentName ?? this.profile.environment,
    );
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
    const maxThrottleWaitSecs = this.profile.maxThrottleWaitSecs;
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
        // Don't retry streaming calls, or if retries are disabled. The signal
        // still has to go with them: it is what cancels a call whose iterator
        // nobody is pumping, and dropping it here leaves the caller holding a
        // signal that reaches nothing.
        return yield* call.next(call.request, { ...restOptions, signal });
      }

      const retryableCodes = new Set([
        ...retryableGrpcStatusCodes,
        ...additionalStatusCodes,
      ]);

      // One idempotency key for the whole call (all attempts).
      const idempotencyKey = uuidv4();

      // Determine max throttle wait: option > MODAL_MAX_THROTTLE_WAIT env var > null (unlimited).
      // 0 disables server-directed retries entirely; null means unlimited.
      const throttleEnabled = maxThrottleWaitSecs !== 0;

      const startTime = Date.now();
      let attempt = 0;
      let delayMs = baseDelay;
      let throttleRetries = 0;
      let lastServerRetryWarnTime = 0;

      logger.debug("Sending gRPC request", "method", call.method.path);

      while (true) {
        // Clone/augment metadata for this attempt.
        const metadata = new Metadata(restOptions.metadata ?? {});

        metadata.set("x-idempotency-key", idempotencyKey);
        metadata.set("x-retry-attempt", String(attempt));
        metadata.set("x-throttle-retry-attempt", String(throttleRetries));
        if (attempt > 0) {
          metadata.set(
            "x-retry-delay",
            ((Date.now() - startTime) / 1000).toFixed(3),
          );
        }
        if (throttleRetries > 0) {
          metadata.set(
            "x-throttle-retry-delay",
            ((Date.now() - startTime) / 1000).toFixed(3),
          );
        }

        // Capture grpc-status-details-bin for server retry policy extraction.
        let capturedStatusDetails: Uint8Array | undefined;
        const onTrailer = (trailer: Metadata) => {
          const raw = trailer.get("grpc-status-details-bin");
          if (raw != null) capturedStatusDetails = raw as Uint8Array;
          restOptions.onTrailer?.(trailer);
        };

        try {
          // Forward the call.
          return yield* call.next(call.request, {
            ...restOptions,
            metadata,
            signal,
            onTrailer,
          });
        } catch (err) {
          // Check for server-directed retry via RPCRetryPolicy in error details.
          // These are handled independently of the client retry counter.
          // maxThrottleWaitSecs === 0 disables server-directed retries entirely.
          const serverPolicy = getServerRetryPolicy(capturedStatusDetails);
          if (serverPolicy && throttleEnabled) {
            const serverDelayMs = Math.max(
              serverPolicy.retryAfterSecs * 1000,
              baseDelay,
            );
            const serverDelaySecs = serverDelayMs / 1000;
            const elapsedSecs = (Date.now() - startTime) / 1000;

            // If max throttle wait is set, stop retrying once the cumulative elapsed
            // time plus the next server delay would exceed the limit.
            if (
              maxThrottleWaitSecs &&
              elapsedSecs + serverDelaySecs >= maxThrottleWaitSecs
            ) {
              logger.debug(
                "Max throttle wait exceeded, not retrying",
                "method",
                call.method.path,
                "elapsed_secs",
                elapsedSecs,
                "server_delay_secs",
                serverDelaySecs,
                "throttle_retries",
                throttleRetries,
                "idempotency_key",
                idempotencyKey.substring(0, 8),
              );
              throw err;
            }

            logger.debug(
              "Server requested retry delay",
              "method",
              call.method.path,
              "elapsed_secs",
              elapsedSecs,
              "server_delay_secs",
              serverDelaySecs,
              "throttle_retries",
              throttleRetries,
              "idempotency_key",
              idempotencyKey.substring(0, 8),
            );

            const now = Date.now();
            if (
              !lastServerRetryWarnTime ||
              now - lastServerRetryWarnTime >= SERVER_RETRY_WARNING_INTERVAL_MS
            ) {
              lastServerRetryWarnTime = now;
              const clientError: ClientError | null =
                err instanceof ClientError ? err : null;
              logger.warn(
                "Server requested retry delay. Retrying...",
                "status",
                clientError?.code ?? "unknown",
                "message",
                clientError?.details ?? String(err),
                "method",
                call.method.path,
              );
            }

            throttleRetries++;
            await sleep(serverDelayMs, signal);
            continue;
          }

          // Immediately propagate non-retryable situations.
          const clientError: ClientError | null =
            err instanceof ClientError ? err : null;
          if (
            !clientError ||
            !retryableCodes.has(clientError.code) ||
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
    const oauthJwtKey = this.oauthJwtKey;

    return async function* authMiddleware<Request, Response>(
      call: ClientMiddlewareCall<Request, Response>,
      options: CallOptions,
    ) {
      const hasOAuthCredentials = Boolean(
        profile.oauthRefreshToken &&
          profile.oauthClientId &&
          (profile.oauthClientSecret || oauthJwtKey),
      );
      if (!hasOAuthCredentials && (!profile.tokenId || !profile.tokenSecret)) {
        throw new Error(
          `Profile is missing credentials. Please set them in .modal.toml, as environment variables, or via the ModalClient constructor.`,
        );
      }

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
      if (hasOAuthCredentials) {
        options.metadata.set(
          "x-modal-refresh-token",
          profile.oauthRefreshToken!,
        );
        options.metadata.set("x-modal-oauth-client-id", profile.oauthClientId!);
        if (oauthJwtKey) {
          options.metadata.set(
            "x-modal-oauth-client-assertion",
            mintOAuthClientAssertion(profile.oauthClientId!, oauthJwtKey),
          );
        } else {
          options.metadata.set(
            "x-modal-oauth-client-secret",
            profile.oauthClientSecret!,
          );
        }
      } else {
        options.metadata.set("x-modal-token-id", profile.tokenId!);
        options.metadata.set("x-modal-token-secret", profile.tokenSecret!);
      }

      // Skip auth token for AuthTokenGet requests to prevent it from getting stuck
      if (call.method.path !== "/modal.client.ModalClient/AuthTokenGet") {
        const tokenManager = getOrCreateAuthTokenManager();
        // getToken() will automatically wait if initial fetch is in progress
        const token = await tokenManager.getToken();
        if (token) {
          options.metadata.set("x-modal-auth-token", token);
        }
      }

      options.signal?.throwIfAborted();
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

const SERVER_RETRY_WARNING_INTERVAL_MS = 30_000;

/** Extract RPCRetryPolicy from the grpc-status-details-bin trailer bytes, or null if absent. */
export function getServerRetryPolicy(
  statusDetails: Uint8Array | undefined,
): RPCRetryPolicy | null {
  if (!statusDetails) return null;
  try {
    const rpcStatus = RPCStatus.decode(statusDetails);
    for (const detail of rpcStatus.details) {
      if (detail.typeUrl.endsWith("/modal.client.RPCRetryPolicy")) {
        return RPCRetryPolicy.decode(detail.value);
      }
    }
  } catch {
    // Ignore decode errors; server may send unexpected format.
  }
  return null;
}

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

export type RetryOptions = {
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
