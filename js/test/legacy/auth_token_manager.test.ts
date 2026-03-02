import { describe, test, expect, vi, beforeEach } from "vitest";
import jwt from "jsonwebtoken";
import { AuthTokenManager } from "../../src/auth_token_manager";
import { newLogger } from "../../src/logger";

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

    const firstToken = await manager.getToken();
    expect(firstToken).toBe(token);

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

    const token = await manager.getToken();
    expect(token).toBe(freshToken);
  });

  test("TestAuthToken_RefreshNearExpiryToken", async () => {
    const now = Math.floor(Date.now() / 1000);
    const expiringToken = createTestJWT(now + 60);
    const freshToken = createTestJWT(now + 3600);

    manager.setToken(expiringToken, now + 60);
    mockClient.setAuthToken(freshToken);

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

    // authTokenGet should have been called only once (during start)
    expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);
  });

  test("TestAuthToken_ProactiveRefreshFailureReturnsOldToken", async () => {
    const now = Math.floor(Date.now() / 1000);
    const nearExpiryToken = createTestJWT(now + 60);
    manager.setToken(nearExpiryToken, now + 60);

    mockClient.authTokenGet.mockRejectedValueOnce(new Error("server blip"));

    const token = await manager.getToken();
    expect(token).toBe(nearExpiryToken);
    expect(mockClient.authTokenGet).toHaveBeenCalledTimes(1);
  });

  test("TestAuthToken_GetToken_EmptyResponse", async () => {
    await expect(manager.getToken()).rejects.toThrow(
      "did not receive auth token from server",
    );
  });
});
