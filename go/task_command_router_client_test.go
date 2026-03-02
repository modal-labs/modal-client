package modal

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func mockJWT(exp any) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	var payloadJSON []byte
	if exp != nil {
		payloadJSON, _ = json.Marshal(map[string]any{"exp": exp})
	} else {
		payloadJSON, _ = json.Marshal(map[string]any{})
	}
	payload := base64.RawURLEncoding.EncodeToString(payloadJSON)
	signature := "fake-signature"
	return header + "." + payload + "." + signature
}

func TestParseJwtExpirationWithValidJWT(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	exp := time.Now().Unix() + 3600
	jwt := mockJWT(exp)
	result, err := parseJwtExpiration(jwt)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(result).ToNot(gomega.BeNil())
	g.Expect(*result).To(gomega.Equal(exp))
}

func TestParseJwtExpirationWithoutExpClaim(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	jwt := mockJWT(nil)
	result, err := parseJwtExpiration(jwt)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.BeNil())
}

func TestParseJwtExpirationWithMalformedJWT(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	jwt := "only.two"
	result, err := parseJwtExpiration(jwt)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(result).To(gomega.BeNil())
}

func TestParseJwtExpirationWithInvalidBase64(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	jwt := "invalid.!!!invalid!!!.signature"
	result, err := parseJwtExpiration(jwt)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(result).To(gomega.BeNil())
}

func TestParseJwtExpirationWithNonNumericExp(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	jwt := mockJWT("not-a-number")
	result, err := parseJwtExpiration(jwt)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(result).To(gomega.BeNil())
}

func TestCallWithRetriesOnTransientErrorsSuccessOnFirstAttempt(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	callCount := 0
	result, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
		callCount++
		output := "success"
		return &output, nil
	}, defaultRetryOptions(), nil)

	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(*result).To(gomega.Equal("success"))
	g.Expect(callCount).To(gomega.Equal(1))
}

func TestCallWithRetriesOnTransientErrorsRetriesOnTransientCodes(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name    string
		code    codes.Code
		message string
	}{
		{"DeadlineExceeded", codes.DeadlineExceeded, "timeout"},
		{"Unavailable", codes.Unavailable, "unavailable"},
		{"Canceled", codes.Canceled, "cancelled"},
		{"Internal", codes.Internal, "internal error"},
		{"Unknown", codes.Unknown, "unknown error"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			ctx := t.Context()
			callCount := 0
			result, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
				callCount++
				var output string
				if callCount == 1 {
					output = ""
					return &output, status.Error(tc.code, tc.message)
				}
				output = "success"
				return &output, nil
			}, retryOptions{BaseDelay: time.Millisecond, DelayFactor: 1, MaxRetries: intPtr(10)}, nil)

			g.Expect(err).ToNot(gomega.HaveOccurred())
			g.Expect(*result).To(gomega.Equal("success"))
			g.Expect(callCount).To(gomega.Equal(2))
		})
	}
}

func TestCallWithRetriesOnTransientErrorsNonRetryableError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	callCount := 0
	_, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
		callCount++
		return nil, status.Error(codes.InvalidArgument, "invalid")
	}, retryOptions{BaseDelay: time.Millisecond, DelayFactor: 1, MaxRetries: intPtr(10)}, nil)

	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(1))
}

func TestCallWithRetriesOnTransientErrorsMaxRetriesExceeded(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	callCount := 0
	maxRetries := 3
	_, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
		callCount++
		return nil, status.Error(codes.Unavailable, "unavailable")
	}, retryOptions{BaseDelay: time.Millisecond, DelayFactor: 1, MaxRetries: &maxRetries}, nil)

	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(maxRetries + 1))
}

func TestCallWithRetriesOnTransientErrorsDeadlineExceeded(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	callCount := 0
	deadline := time.Now().Add(50 * time.Millisecond)
	_, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
		callCount++
		return nil, status.Error(codes.Unavailable, "unavailable")
	}, retryOptions{BaseDelay: 100 * time.Millisecond, DelayFactor: 1, MaxRetries: nil, Deadline: &deadline}, nil)

	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.Equal("deadline exceeded"))
}

func TestCallWithRetriesOnTransientErrorClosed(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	callCount := 0
	var closed atomic.Bool
	closed.Store(true)

	_, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
		callCount++
		return nil, status.Error(codes.Canceled, "invalid")
	}, defaultRetryOptions(), &closed)

	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))

}

func intPtr(i int) *int {
	return &i
}

type mockRetryableClient struct {
	refreshJwtCallCount  int
	authContextCallCount int
}

func (m *mockRetryableClient) authContext(ctx context.Context) context.Context {
	m.authContextCallCount += 1
	return ctx
}

func (m *mockRetryableClient) refreshJwt(ctx context.Context) error {
	m.refreshJwtCallCount += 1
	return nil
}

func newMockRetryableClient() *mockRetryableClient {
	return &mockRetryableClient{refreshJwtCallCount: 0, authContextCallCount: 0}
}

func TestCallWithAuthRetrySuccessFirstAttempt(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	c := newMockRetryableClient()
	result, err := callWithAuthRetry(ctx, c, func(authCtx context.Context) (*int, error) {
		return intPtr(3), nil
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(c.authContextCallCount).To(gomega.Equal(1))
	g.Expect(c.refreshJwtCallCount).To(gomega.Equal(0))

	g.Expect(result).ToNot(gomega.BeNil())
	g.Expect(*result).To(gomega.Equal(3))
}

func TestCallWithAuthRetryOnUNAUTHENTICATED(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	callCount := 0

	c := newMockRetryableClient()
	result, err := callWithAuthRetry(ctx, c, func(authCtx context.Context) (*int, error) {
		if callCount == 0 {
			callCount += 1
			return nil, status.Error(codes.Unauthenticated, "Not authenticated")
		}
		return intPtr(3), nil
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(c.authContextCallCount).To(gomega.Equal(2))
	g.Expect(c.refreshJwtCallCount).To(gomega.Equal(1))

	g.Expect(result).ToNot(gomega.BeNil())
	g.Expect(*result).To(gomega.Equal(3))

}

func TestCallWithAuthRetryDoesNotRetryOnNonUNAUTHENTICATED(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	c := newMockRetryableClient()
	_, err := callWithAuthRetry(ctx, c, func(authCtx context.Context) (*int, error) {
		return nil, status.Error(codes.InvalidArgument, "Invalid argument")
	})
	g.Expect(err).To(gomega.HaveOccurred())

	g.Expect(c.authContextCallCount).To(gomega.Equal(1))
	g.Expect(c.refreshJwtCallCount).To(gomega.Equal(0))
}

func TestCallWithAuthRetryDoesNotRetryErrorIfUNAUTHENTICATEDAfterRetry(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	c := newMockRetryableClient()
	_, err := callWithAuthRetry(ctx, c, func(authCtx context.Context) (*int, error) {
		return nil, status.Error(codes.Unauthenticated, "Not authenticated")
	})
	g.Expect(err).To(gomega.HaveOccurred())

	g.Expect(c.authContextCallCount).To(gomega.Equal(2))
	g.Expect(c.refreshJwtCallCount).To(gomega.Equal(1))
}
