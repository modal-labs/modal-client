import { describe, it, expect, vi } from "vitest";
import { _retryHttpRequest as retryHttpRequest } from "./blob";

describe("retryHttpRequest", () => {
  it("succeeds on first attempt", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    const result = await retryHttpRequest(fn, 3, 1);
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries on failure and eventually succeeds", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("fail 1"))
      .mockRejectedValueOnce(new Error("fail 2"))
      .mockResolvedValue("ok");
    const result = await retryHttpRequest(fn, 3, 1);
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("throws after exhausting all attempts", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("persistent error"));
    await expect(retryHttpRequest(fn, 3, 1)).rejects.toThrow(
      "persistent error",
    );
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("respects custom attempt count", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("error"));
    await expect(retryHttpRequest(fn, 5, 1)).rejects.toThrow("error");
    expect(fn).toHaveBeenCalledTimes(5);
  });

  it("throws the last error when all attempts fail", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("error 1"))
      .mockRejectedValueOnce(new Error("error 2"))
      .mockRejectedValueOnce(new Error("error 3"));
    await expect(retryHttpRequest(fn, 3, 1)).rejects.toThrow("error 3");
  });

  it("uses exponential backoff between retries", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("fail"))
      .mockResolvedValue("ok");

    const promise = retryHttpRequest(fn, 3, 300);
    await vi.advanceTimersByTimeAsync(0);
    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(300);
    expect(fn).toHaveBeenCalledTimes(2);

    const result = await promise;
    expect(result).toBe("ok");
    vi.useRealTimers();
  });
});
