import { describe, it, expect, vi, test } from "vitest";
import {
  parseLogLevel,
  DefaultLogger,
  createLogger,
  type Logger,
} from "./logger";

test("parseLogLevel", () => {
  expect(parseLogLevel("debug")).toBe("debug");
  expect(parseLogLevel("DEBUG")).toBe("debug");
  expect(parseLogLevel("warning")).toBe("warn");
  expect(parseLogLevel("WARNING")).toBe("warn");

  expect(parseLogLevel("")).toBe("warn");

  expect(() => parseLogLevel("invalid")).toThrow(
    'Invalid log level value: "invalid" (must be debug, info, warn, or error)',
  );
});

describe("createLogger", () => {
  it("should return DefaultLogger when no custom logger provided", () => {
    const logger = createLogger(undefined, "debug");
    expect(logger).toBeInstanceOf(DefaultLogger);
  });

  it("should return FilteredLogger when custom logger provided", () => {
    const mockLogger: Logger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    };

    const logger = createLogger(mockLogger, "debug");
    expect(logger).toBeDefined();
    expect(logger).not.toBeInstanceOf(DefaultLogger);
  });

  it("should apply level filtering to custom logger", () => {
    const mockLogger: Logger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    };

    const logger = createLogger(mockLogger, "warn");

    logger.debug("test");
    logger.info("test");
    logger.warn("test");
    logger.error("test");

    expect(mockLogger.debug).not.toHaveBeenCalled();
    expect(mockLogger.info).not.toHaveBeenCalled();
    expect(mockLogger.warn).toHaveBeenCalledWith("test");
    expect(mockLogger.error).toHaveBeenCalledWith("test");
  });

  it("should pass arguments to custom logger", () => {
    const mockLogger: Logger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    };

    const logger = createLogger(mockLogger, "debug");

    logger.debug("message", "key1", "value1", "key2", 123);

    expect(mockLogger.debug).toHaveBeenCalledWith(
      "message",
      "key1",
      "value1",
      "key2",
      123,
    );
  });
});
