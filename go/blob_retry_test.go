package modal

import (
	"context"
	"fmt"
	"log/slog"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

func TestRetryHTTPRequestSuccessOnFirstAttempt(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	logger := slog.Default()

	callCount := 0
	result, err := retryHTTPRequest(ctx, logger, "test", 3, time.Millisecond, func() (string, error) {
		callCount++
		return "ok", nil
	})

	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal("ok"))
	g.Expect(callCount).To(gomega.Equal(1))
}

func TestRetryHTTPRequestSuccessAfterRetries(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	logger := slog.Default()

	callCount := 0
	result, err := retryHTTPRequest(ctx, logger, "test", 3, time.Millisecond, func() (string, error) {
		callCount++
		if callCount < 3 {
			return "", fmt.Errorf("transient error")
		}
		return "ok", nil
	})

	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal("ok"))
	g.Expect(callCount).To(gomega.Equal(3))
}

func TestRetryHTTPRequestExhaustsAttempts(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	logger := slog.Default()

	callCount := 0
	result, err := retryHTTPRequest(ctx, logger, "test", 5, time.Millisecond, func() (string, error) {
		callCount++
		return "", fmt.Errorf("persistent error")
	})

	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.Equal("persistent error"))
	g.Expect(result).To(gomega.Equal(""))
	g.Expect(callCount).To(gomega.Equal(5))
}

func TestRetryHTTPRequestRespectsContextCancellation(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx, cancel := context.WithCancel(t.Context())
	logger := slog.Default()

	callCount := 0
	_, err := retryHTTPRequest(ctx, logger, "test", 3, time.Millisecond, func() (string, error) {
		callCount++
		cancel()
		return "", fmt.Errorf("error triggering retry")
	})

	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(1))
}
