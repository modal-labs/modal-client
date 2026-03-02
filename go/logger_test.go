package modal

import (
	"log/slog"
	"testing"

	"github.com/onsi/gomega"
)

func TestParseLogLevel(t *testing.T) {
	g := gomega.NewWithT(t)

	tests := []struct {
		input    string
		expected slog.Level
	}{
		{"DEBUG", slog.LevelDebug},
		{"INFO", slog.LevelInfo},
		{"WARN", slog.LevelWarn},
		{"WARNING", slog.LevelWarn},
		{"ERROR", slog.LevelError},
		{"eRrOr", slog.LevelError},
		{"", slog.LevelWarn},
	}

	for _, tt := range tests {
		level, err := parseLogLevel(tt.input)
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		g.Expect(level).Should(gomega.Equal(tt.expected))
	}
}

func TestParseLogLevel_InvalidValue(t *testing.T) {
	g := gomega.NewWithT(t)

	level, err := parseLogLevel("invalid")
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("invalid log level"))
	g.Expect(level).Should(gomega.Equal(slog.LevelWarn))
}
