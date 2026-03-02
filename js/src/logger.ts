/* eslint-disable no-console */
export type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

export interface Logger {
  debug(message: string, ...args: any[]): void;
  info(message: string, ...args: any[]): void;
  warn(message: string, ...args: any[]): void;
  error(message: string, ...args: any[]): void;
}

export function parseLogLevel(level: string): LogLevel {
  if (!level) {
    return "warn";
  }

  const normalized = level.toLowerCase();
  if (
    normalized === "debug" ||
    normalized === "info" ||
    normalized === "warn" ||
    normalized === "warning" ||
    normalized === "error"
  ) {
    return normalized === "warning" ? "warn" : (normalized as LogLevel);
  }

  throw new Error(
    `Invalid log level value: "${level}" (must be debug, info, warn, or error)`,
  );
}

export class DefaultLogger implements Logger {
  private levelValue: number;

  constructor(level: LogLevel = "warn") {
    this.levelValue = LOG_LEVELS[level];
  }

  debug(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.debug) {
      console.log(this.formatMessage("DEBUG", message, args));
    }
  }

  info(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.info) {
      console.log(this.formatMessage("INFO", message, args));
    }
  }

  warn(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.warn) {
      console.warn(this.formatMessage("WARN", message, args));
    }
  }

  error(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.error) {
      console.error(this.formatMessage("ERROR", message, args));
    }
  }

  private formatMessage(level: string, message: string, args: any[]): string {
    const timestamp = new Date().toISOString();
    let formatted = `time=${timestamp} level=${level} msg="${message}"`;

    if (args.length > 0) {
      for (let i = 0; i < args.length; i += 2) {
        if (i + 1 < args.length) {
          const key = args[i];
          const value = args[i + 1];
          formatted += ` ${key}=${this.formatValue(value)}`;
        }
      }
    }

    return formatted;
  }

  private formatValue(value: any): string {
    if (typeof value === "string") {
      return value.includes(" ") ? `"${value}"` : value;
    }
    if (value instanceof Error) {
      return `"${value.message}"`;
    }
    if (Array.isArray(value)) {
      return `[${value.join(",")}]`;
    }
    return String(value);
  }
}

class FilteredLogger implements Logger {
  private levelValue: number;

  constructor(
    private logger: Logger,
    level: LogLevel,
  ) {
    this.levelValue = LOG_LEVELS[level];
  }

  debug(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.debug) {
      this.logger.debug(message, ...args);
    }
  }

  info(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.info) {
      this.logger.info(message, ...args);
    }
  }

  warn(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.warn) {
      this.logger.warn(message, ...args);
    }
  }

  error(message: string, ...args: any[]): void {
    if (this.levelValue <= LOG_LEVELS.error) {
      this.logger.error(message, ...args);
    }
  }
}

export function createLogger(logger?: Logger, logLevel: string = ""): Logger {
  const level = parseLogLevel(logLevel);

  if (logger) {
    return new FilteredLogger(logger, level);
  }

  return new DefaultLogger(level);
}

export function newLogger(logLevel: string = ""): Logger {
  return createLogger(undefined, logLevel);
}
