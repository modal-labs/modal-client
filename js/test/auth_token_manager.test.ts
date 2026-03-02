import { describe, test, expect, vi, beforeEach } from "vitest";
import jwt from "jsonwebtoken";
import { ModalClient } from "../src/client";
import { AuthTokenManager, REFRESH_WINDOW } from "../src/auth_token_manager";
import { newLogger } from "../src/logger";

class mockAuthClient {
  private authToken: string = "";

  setAuthToken(token: string) {
    this.authToken = token;
  }

  authTokenGet = vi.fn(async () => {
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

  test("TestAuthToken_GetToken_EmptyResponse", async () => {
    // authToken is "" by default, so authTokenGet returns empty
    await expect(manager.getToken()).rejects.toThrow(
      "did not receive auth token from server",
    );
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
