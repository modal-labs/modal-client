package modal

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
)

func parseLogLevel(level string) (slog.Level, error) {
	if level == "" {
		return slog.LevelWarn, nil
	}

	switch strings.ToUpper(level) {
	case "DEBUG":
		return slog.LevelDebug, nil
	case "INFO":
		return slog.LevelInfo, nil
	case "WARN", "WARNING":
		return slog.LevelWarn, nil
	case "ERROR":
		return slog.LevelError, nil
	default:
		return slog.LevelWarn, fmt.Errorf("invalid log level value: %q (must be DEBUG, INFO, WARN, or ERROR)", level)
	}
}

func newLogger(profile Profile) (*slog.Logger, error) {
	level, err := parseLogLevel(profile.LogLevel)
	if err != nil {
		return nil, err
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
	})
	return slog.New(handler), nil
}
