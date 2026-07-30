import { ClientError, Status } from "nice-grpc";
import type { Logger } from "./logger";

// Start refreshing this many seconds before the token expires
export const REFRESH_WINDOW = 5 * 60;
export const AUTH_TOKEN_GET_TIMEOUT_MS = 5 * 1000;
export const AUTH_TOKEN_GET_MAX_RETRIES = 3;
// After a failed refresh, wait before hitting the server again, growing exponentially with
// consecutive failures between these bounds (in milliseconds).
export const FAILURE_BACKOFF_BASE_MS = 500;
export const FAILURE_BACKOFF_MAX_MS = 60 * 1000;
// If the token doesn't have an expiry field, default to current time plus this value (not expected).
export const DEFAULT_EXPIRY_OFFSET = 20 * 60;

/**
 * Manages authentication tokens, refreshing them lazily when getToken is
 * called. Tokens are refreshed when expired or within REFRESH_WINDOW seconds
 * of expiry.
 *
 * Three states:
 *  1. Valid token (not near expiry): returned immediately.
 *  2. No token or expired: all callers block until a fresh token is fetched.
 *     Only one fetch happens; concurrent callers await the same promise.
 *  3. Valid but within REFRESH_WINDOW of expiry: if no refresh is in progress
 *     the caller triggers one (blocking itself); concurrent callers get the
 *     old, still-valid token.
 */
export class AuthTokenManager {
  private client: any;
  private logger: Logger;
  private currentToken: string = "";
  private tokenExpiry: number = 0;
  private refreshPromise: Promise<void> | null = null;
  private retryAfter: number = 0;
  private backoffMs: number = FAILURE_BACKOFF_BASE_MS;

  constructor(client: any, logger: Logger) {
    this.client = client;
    this.logger = logger;
  }

  /**
   * Returns a valid auth token.
   */
  async getToken(): Promise<string> {
    if (!this.currentToken || this.isExpired()) {
      return this.tryRefreshToken();
    }

    if (this.shouldRefresh() && !this.refreshPromise) {
      await this.tryRefreshToken();
    }

    return this.currentToken;
  }

  private async tryRefreshToken(): Promise<string> {
    try {
      return await this.lockedRefreshToken();
    } catch (error) {
      if (this.isAuthDenied(error)) {
        throw error;
      }
      if (!this.currentToken) {
        throw error;
      }
      this.logger.warn(
        "Auth token refresh failed; falling back to cached token",
        "error",
        error,
      );
      return this.currentToken;
    }
  }

  /**
   * Ensures only one fetch is in progress at a time. Concurrent callers
   * await the same promise. Includes a double-check so that if another
   * caller already refreshed, we skip the RPC.
   */
  private async lockedRefreshToken(): Promise<string> {
    if (this.currentToken && !this.shouldRefresh()) {
      return this.currentToken;
    }
    if (!this.refreshPromise) {
      this.refreshPromise = (async () => {
        try {
          await this.fetchToken();
        } finally {
          this.refreshPromise = null;
        }
      })();
    }
    await this.refreshPromise;
    return this.currentToken;
  }

  /**
   * Fetches a new auth token from the server and stores it.
   */
  private async fetchToken(): Promise<void> {
    try {
      const hasCachedToken = !!this.currentToken;
      const response = await this.client.authTokenGet(
        {},
        {
          // No cached token to fall back on, so a failure is user-visible: retry transient errors.
          // Otherwise one attempt, and retryAfter handles the cooldown.
          retries: hasCachedToken ? 0 : AUTH_TOKEN_GET_MAX_RETRIES,
          timeoutMs: AUTH_TOKEN_GET_TIMEOUT_MS,
        },
      );
      const token = response.token;
      if (!token) {
        throw new Error(
          "Internal error: did not receive auth token from server, please contact Modal support",
        );
      }
      this.currentToken = token;
      const exp = this.decodeJWT(token);
      if (exp > 0) {
        this.tokenExpiry = exp;
      } else {
        this.logger.warn("x-modal-auth-token does not contain exp field");
        // We'll use the token, and set the expiry to DEFAULT_EXPIRY_OFFSET from now.
        this.tokenExpiry =
          Math.floor(Date.now() / 1000) + DEFAULT_EXPIRY_OFFSET;
      }
      this.retryAfter = 0;
      this.backoffMs = FAILURE_BACKOFF_BASE_MS;

      const now = Math.floor(Date.now() / 1000);
      const expiresIn = this.tokenExpiry - now;
      const refreshIn = this.tokenExpiry - now - REFRESH_WINDOW;
      this.logger.debug(
        "Fetched auth token",
        "expires_in",
        `${expiresIn}s`,
        "refresh_in",
        `${refreshIn}s`,
      );
    } catch (error) {
      if (this.isAuthDenied(error)) {
        throw error;
      }
      // Back off (exponentially on consecutive failures) so we don't hammer a struggling server.
      this.retryAfter = Date.now() + this.backoffMs;
      this.backoffMs = Math.min(this.backoffMs * 2, FAILURE_BACKOFF_MAX_MS);
      throw error;
    }
  }

  /**
   * Extracts the exp claim from a JWT token.
   */
  private decodeJWT(token: string): number {
    try {
      const parts = token.split(".");
      if (parts.length !== 3) {
        return 0;
      }

      let payload = parts[1];
      while (payload.length % 4 !== 0) {
        payload += "=";
      }

      const decoded = atob(payload);
      const claims = JSON.parse(decoded);
      return claims.exp || 0;
    } catch {
      return 0;
    }
  }

  isExpired(): boolean {
    const now = Math.floor(Date.now() / 1000);
    return now >= this.tokenExpiry;
  }

  private needsRefresh(): boolean {
    const now = Math.floor(Date.now() / 1000);
    return now >= this.tokenExpiry - REFRESH_WINDOW;
  }

  private inBackoff(): boolean {
    return Date.now() < this.retryAfter;
  }

  private isAuthDenied(error: unknown): boolean {
    // The credentials themselves were rejected (e.g. revoked/invalid key), as opposed to a
    // transient outage/overload. Fail fast instead of reusing the cached token or backing off.
    return (
      error instanceof ClientError &&
      (error.code === Status.UNAUTHENTICATED ||
        error.code === Status.PERMISSION_DENIED)
    );
  }

  private shouldRefresh(): boolean {
    return this.needsRefresh() && !this.inBackoff();
  }

  getCurrentToken(): string {
    return this.currentToken;
  }

  setToken(token: string, expiry: number): void {
    this.currentToken = token;
    this.tokenExpiry = expiry;
  }
}
