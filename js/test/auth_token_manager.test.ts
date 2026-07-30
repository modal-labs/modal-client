import { describe, test, expect, vi, beforeEach } from "vitest";
import jwt from "jsonwebtoken";
import { ClientError, Status } from "nice-grpc";
import { ModalClient } from "../src/client";
import {
  AUTH_TOKEN_GET_TIMEOUT_MS,
  AUTH_TOKEN_GET_MAX_RETRIES,
  AuthTokenManager,
  FAILURE_BACKOFF_BASE_MS,
  FAILURE_BACKOFF_MAX_MS,
  REFRESH_WINDOW,
} from "../src/auth_token_manager";
import { newLogger } from "../src/logger";

class mockAuthClient {
  private authToken: string = "";

  setAuthToken(token: string) {
    this.authToken = token;
  }

  authTokenGet = vi.fn(async (_request?: unknown, _options?: unknown) => {
    return { token: this.authToken };
  });
}

function newMockAuthClient() {
  return new mockAuthClient();
}

// Creates a JWT token for testing
function createTestJWT(expiry: number): string {
  return jwt.sign({ exp: expiry }, "walter-test");
}

describe("AuthTokenManager", () => {
  let mockClient: mockAuthClient;
  let manager: AuthTokenManager;

  beforeEach(() => {
    mockClient = newMockAuthClient();
    manager = new AuthTokenManager(mockClient as any, newLogger());
  });

  test("TestAuthToken_DecodeJWT", async () => {
    const now = Math.floor(Date.now() / 1000);
    const expiry = now + 1800;
    const token = createTestJWT(expiry);
    mockClient.setAuthToken(token);

    const result = await manager.getToken();
    expect(result).toBe(token);
    expect(manager.getCurrentToken()).toBe(token);
  });

  test("TestAuthToken_LazyFetch", async () => {
    const now = Math.floor(Date.now() / 1000);
    const token = createTestJWT(now + 3600);
    mockClient.setAuthToken(token);

    // First getToken lazily fetches
    const firstToken = await manager.getToken();
    expect(firstToken).toBe(token);

    // Second getToken returns cached
    const secondToken = await manager.getToken();
    expect(secondToken).toBe(token);

    expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);
  });

  test("TestAuthToken_FetchOnlyRetriesWithoutCachedToken", async () => {
    const token = createTestJWT(Math.floor(Date.now() / 1000) + 3600);
    mockClient.setAuthToken(token);

    await expect(manager.getToken()).resolves.toBe(token);
    expect(mockClient.authTokenGet.mock.calls[0][1]).toEqual({
      retries: AUTH_TOKEN_GET_MAX_RETRIES,
      timeoutMs: AUTH_TOKEN_GET_TIMEOUT_MS,
    });

    const expiredToken = createTestJWT(Math.floor(Date.now() / 1000) - 60);
    manager.setToken(expiredToken, Math.floor(Date.now() / 1000) - 60);
    await expect(manager.getToken()).resolves.toBe(token);
    expect(mockClient.authTokenGet.mock.calls[1][1]).toEqual({
      retries: 0,
      timeoutMs: AUTH_TOKEN_GET_TIMEOUT_MS,
    });
  });

  test("TestAuthToken_IsExpired", async () => {
    const now = Math.floor(Date.now() / 1000);

    // Test not expired
    const validToken = createTestJWT(now + 3600);
    manager.setToken(validToken, now + 3600);
    expect(manager.isExpired()).toBe(false);

    // Test expired
    const expiredToken = createTestJWT(now - 60);
    manager.setToken(expiredToken, now - 60);
    expect(manager.isExpired()).toBe(true);
  });

  test("TestAuthToken_RefreshExpiredToken", async () => {
    const now = Math.floor(Date.now() / 1000);
    const expiringToken = createTestJWT(now - 60);
    const freshToken = createTestJWT(now + 3600);

    manager.setToken(expiringToken, now - 60);
    mockClient.setAuthToken(freshToken);

    // getToken should see the expired token and fetch a new one
    const token = await manager.getToken();
    expect(token).toBe(freshToken);
  });

  test("TestAuthToken_RefreshNearExpiryToken", async () => {
    const now = Math.floor(Date.now() / 1000);
    // Token within REFRESH_WINDOW of expiry (60s left, window is 300s)
    const expiringToken = createTestJWT(now + 60);
    const freshToken = createTestJWT(now + 3600);

    manager.setToken(expiringToken, now + 60);
    mockClient.setAuthToken(freshToken);

    // getToken should proactively refresh
    const token = await manager.getToken();
    expect(token).toBe(freshToken);
  });

  test("TestAuthToken_ConcurrentGetToken", async () => {
    const token = createTestJWT(Math.floor(Date.now() / 1000) + 3600);
    mockClient.setAuthToken(token);

    // Multiple concurrent getToken calls should all return the same token
    const [result1, result2, result3] = await Promise.all([
      manager.getToken(),
      manager.getToken(),
      manager.getToken(),
    ]);
    expect(result1).toBe(token);
    expect(result2).toBe(token);
    expect(result3).toBe(token);

    // Only one fetch should have happened
    expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);
  });

  test("TestAuthToken_ConcurrentGetTokenWithExpiredToken", async () => {
    const now = Math.floor(Date.now() / 1000);

    const expiredToken = createTestJWT(now - 10);
    manager.setToken(expiredToken, now - 10);

    const freshToken = createTestJWT(now + 3600);
    mockClient.setAuthToken(freshToken);

    const [result1, result2, result3] = await Promise.all([
      manager.getToken(),
      manager.getToken(),
      manager.getToken(),
    ]);

    expect(result1).toBe(freshToken);
    expect(result2).toBe(freshToken);
    expect(result3).toBe(freshToken);
    expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);
  });

  test("TestAuthToken_ProactiveRefreshFailureReturnsOldToken", async () => {
    const now = Math.floor(Date.now() / 1000);
    // Token within REFRESH_WINDOW of expiry (60s left, window is 300s)
    const nearExpiryToken = createTestJWT(now + 60);
    manager.setToken(nearExpiryToken, now + 60);

    // Make the refresh RPC fail
    mockClient.authTokenGet.mockRejectedValueOnce(new Error("server blip"));

    // getToken should return the old valid token, not throw
    const token = await manager.getToken();
    expect(token).toBe(nearExpiryToken);
    expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);
  });

  test("TestAuthToken_NoCachedTokenRefreshFailureRejects", async () => {
    mockClient.authTokenGet.mockRejectedValue(new Error("server blip"));

    await expect(manager.getToken()).rejects.toThrow("server blip");
  });

  test("TestAuthToken_ExpiredRefreshBackoffClearsRefreshPromise", async () => {
    vi.useFakeTimers();
    try {
      const baseTime = new Date("2025-01-01T00:00:00Z");
      vi.setSystemTime(baseTime);
      const now = Math.floor(baseTime.getTime() / 1000);
      const expiredToken = createTestJWT(now - 60);
      const refreshedToken = createTestJWT(now + 3600);
      manager.setToken(expiredToken, now - 60);
      mockClient.authTokenGet.mockRejectedValueOnce(new Error("server blip"));

      await expect(manager.getToken()).resolves.toBe(expiredToken);
      await expect(manager.getToken()).resolves.toBe(expiredToken);
      expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);

      mockClient.setAuthToken(refreshedToken);
      vi.advanceTimersByTime(FAILURE_BACKOFF_BASE_MS + 1);

      await expect(manager.getToken()).resolves.toBe(refreshedToken);
      expect(mockClient.authTokenGet).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  test("TestAuthToken_GetToken_EmptyResponse", async () => {
    // authToken is "" by default, so authTokenGet returns empty
    await expect(manager.getToken()).rejects.toThrow(
      "did not receive auth token from server",
    );
  });

  test("TestAuthToken_RefreshBackoffGrowsExponentially", async () => {
    vi.useFakeTimers();
    try {
      const baseTime = new Date("2025-01-01T00:00:00Z");
      vi.setSystemTime(baseTime);
      const now = Math.floor(baseTime.getTime() / 1000);
      const expiredToken = createTestJWT(now - 60);
      const freshToken = createTestJWT(now + 3600);
      manager.setToken(expiredToken, now - 60);
      mockClient.authTokenGet.mockRejectedValue(new Error("server blip"));

      const expectedBackoffs = [
        FAILURE_BACKOFF_BASE_MS,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        FAILURE_BACKOFF_MAX_MS,
        FAILURE_BACKOFF_MAX_MS,
      ];
      for (const expectedBackoff of expectedBackoffs) {
        await expect(manager.getToken()).resolves.toBe(expiredToken);
        expect(manager["retryAfter"]).toBe(Date.now() + expectedBackoff);
        vi.setSystemTime(new Date(Date.now() + expectedBackoff + 1));
      }

      mockClient.setAuthToken(freshToken);
      mockClient.authTokenGet.mockResolvedValue({ token: freshToken });
      await expect(manager.getToken()).resolves.toBe(freshToken);
      expect(manager["backoffMs"]).toBe(FAILURE_BACKOFF_BASE_MS);

      manager.setToken(expiredToken, now - 60);
      mockClient.authTokenGet.mockRejectedValue(new Error("server blip"));
      await expect(manager.getToken()).resolves.toBe(expiredToken);
      expect(manager["retryAfter"]).toBe(Date.now() + FAILURE_BACKOFF_BASE_MS);
    } finally {
      vi.useRealTimers();
    }
  });

  test.each([Status.UNAUTHENTICATED, Status.PERMISSION_DENIED])(
    "TestAuthToken_AuthDeniedDoesNotFallBack (%s)",
    async (status) => {
      const now = Math.floor(Date.now() / 1000);
      const expiredToken = createTestJWT(now - 60);
      manager.setToken(expiredToken, now - 60);
      mockClient.authTokenGet.mockRejectedValue(
        new ClientError("/auth-token", status, "credentials rejected"),
      );

      await expect(manager.getToken()).rejects.toThrow("credentials rejected");
      expect(manager.getCurrentToken()).toBe(expiredToken);
      expect(manager["backoffMs"]).toBe(FAILURE_BACKOFF_BASE_MS);
      expect(manager["inBackoff"]()).toBe(false);
    },
  );

  test("TestAuthToken_TransientFailureFallsBackAndBacksOff", async () => {
    const now = Math.floor(Date.now() / 1000);
    const expiredToken = createTestJWT(now - 60);
    manager.setToken(expiredToken, now - 60);
    mockClient.authTokenGet.mockRejectedValue(
      new ClientError("/auth-token", Status.UNAVAILABLE, "server unavailable"),
    );

    await expect(manager.getToken()).resolves.toBe(expiredToken);
    expect(manager["backoffMs"]).toBe(2 * FAILURE_BACKOFF_BASE_MS);
    expect(manager["inBackoff"]()).toBe(true);
    expect(manager["retryAfter"]).toBeGreaterThan(Date.now());
  });

  test("TestAuthToken_ExpiredThenRefreshed", async () => {
    vi.useFakeTimers();
    try {
      const baseTime = new Date("2025-01-01T00:00:00Z");
      vi.setSystemTime(baseTime);
      const baseTimeSeconds = Math.floor(baseTime.getTime() / 1000);

      const tokenOneExpirySeconds = baseTimeSeconds + REFRESH_WINDOW + 5;

      // First getToken lazily fetches tokenOne
      const tokenOne = createTestJWT(tokenOneExpirySeconds);
      mockClient.setAuthToken(tokenOne);
      await expect(manager.getToken()).resolves.toBe(tokenOne);
      expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);

      // Simulate time moving past token expiry
      const tokenTwo = createTestJWT(tokenOneExpirySeconds + 3600);
      mockClient.setAuthToken(tokenTwo);
      vi.setSystemTime(new Date((tokenOneExpirySeconds + 1) * 1000));

      // getToken should see tokenOne expired and fetch tokenTwo
      await expect(manager.getToken()).resolves.toBe(tokenTwo);
      expect(mockClient.authTokenGet).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("ModalClient with AuthTokenManager", () => {
  test("TestModalClient_CloseCleansUpAuthTokenManager", () => {
    const mockCpClient = newMockAuthClient();
    const client = new ModalClient({
      cpClient: mockCpClient as any,
    });

    client.close();
  });

  test("TestModalClient_MultipleInstancesHaveSeparateManagers", () => {
    const mockCpClient1 = newMockAuthClient();
    const mockCpClient2 = newMockAuthClient();

    const client1 = new ModalClient({
      cpClient: mockCpClient1 as any,
    });

    const client2 = new ModalClient({
      cpClient: mockCpClient2 as any,
    });

    client1.close();
    client2.close();
  });
});
