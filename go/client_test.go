package modal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

func TestClientWithLogger(t *testing.T) {
	g := gomega.NewWithT(t)

	r, w, err := os.Pipe()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	logger := slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: slog.LevelDebug}))
	g.Expect(logger).NotTo(gomega.BeNil())

	client, err := NewClientWithOptions(&ClientParams{Logger: logger})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(client).NotTo(gomega.BeNil())

	err = w.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(r)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(output).To(gomega.ContainSubstring("Initializing Modal client"))
	g.Expect(output).To(gomega.ContainSubstring("Modal client initialized successfully"))
}

func TestClientWithCustomInterceptors(t *testing.T) {
	g := gomega.NewWithT(t)

	var firstCalled, secondCalled bool
	var firstMethod, secondMethod string
	var mu sync.Mutex

	firstInterceptor := func(
		ctx context.Context,
		method string,
		req, reply any,
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		mu.Lock()
		firstCalled = true
		firstMethod = method
		mu.Unlock()
		return invoker(ctx, method, req, reply, cc, opts...)
	}

	secondInterceptor := func(
		ctx context.Context,
		method string,
		req, reply any,
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		mu.Lock()
		secondCalled = true
		secondMethod = method
		mu.Unlock()
		return invoker(ctx, method, req, reply, cc, opts...)
	}

	client, err := NewClientWithOptions(&ClientParams{
		GRPCUnaryInterceptors: []grpc.UnaryClientInterceptor{firstInterceptor, secondInterceptor},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(client).NotTo(gomega.BeNil())
	defer client.Close()

	_, err = client.Functions.FromName(t.Context(), "libmodal-test-support", "non-existent", nil)
	g.Expect(err).Should(gomega.HaveOccurred()) // don't care about success here, just need the RPC

	mu.Lock()
	g.Expect(firstCalled).To(gomega.BeTrue())
	g.Expect(firstMethod).To(gomega.ContainSubstring("ModalClient/"))
	g.Expect(secondCalled).To(gomega.BeTrue())
	g.Expect(secondMethod).To(gomega.ContainSubstring("ModalClient/"))
	mu.Unlock()
}

// makeThrottleError builds a gRPC RESOURCE_EXHAUSTED error carrying an RPCRetryPolicy detail
// using the type.modal.com/ type URL prefix (matching the server-side encoding).
func makeThrottleError(t *testing.T, delaySecs float32) error {
	t.Helper()
	policy := pb.RPCRetryPolicy_builder{RetryAfterSecs: delaySecs}.Build()
	st := status.New(codes.ResourceExhausted, "server throttled")
	withDetails, err := statusWithDetails(st, policy)
	if err != nil {
		t.Fatalf("statusWithDetails: %v", err)
	}
	return withDetails.Err()
}

// statusWithDetails is like status.Status.WithDetails but uses type.modal.com/
// as the type URL prefix instead of the default type.googleapis.com/.
func statusWithDetails(s *status.Status, details ...proto.Message) (*status.Status, error) {
	if s.Code() == codes.OK {
		return nil, fmt.Errorf("no error details for status with code OK")
	}
	p := s.Proto()
	for _, detail := range details {
		b, err := proto.Marshal(detail)
		if err != nil {
			return nil, fmt.Errorf("proto.Marshal: %w", err)
		}
		fullName := string(detail.ProtoReflect().Descriptor().FullName())
		p.Details = append(p.Details, &anypb.Any{
			TypeUrl: "type.modal.com/" + fullName,
			Value:   b,
		})
	}
	return status.FromProto(p), nil
}

func TestGetServerRetryPolicy(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// Non-gRPC error returns nil.
	g.Expect(getServerRetryPolicy(errors.New("plain error"))).To(gomega.BeNil())

	// gRPC error without details returns nil.
	g.Expect(getServerRetryPolicy(status.Error(codes.Unavailable, "down"))).To(gomega.BeNil())

	// gRPC error with RPCRetryPolicy returns the policy.
	throttleErr := makeThrottleError(t, 2.5)
	result := getServerRetryPolicy(throttleErr)
	g.Expect(result).NotTo(gomega.BeNil())
	g.Expect(result.GetRetryAfterSecs()).To(gomega.BeNumerically("~", 2.5, 0.01))
}

func TestRetryInterceptorServerDrivenRetry(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	throttleErr := makeThrottleError(t, 0.01)

	callCount := 0
	var lastMD metadata.MD
	c := &Client{logger: slog.New(slog.DiscardHandler)}
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		lastMD, _ = metadata.FromOutgoingContext(ctx)
		if callCount <= 3 {
			return throttleErr
		}
		return nil
	}

	err := interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(4)) // 3 throttle failures + 1 success

	// Server-driven retries tracked separately; client attempt counter stays at 0.
	g.Expect(lastMD.Get("x-throttle-retry-attempt")).To(gomega.Equal([]string{"3"}))
	g.Expect(lastMD.Get("x-retry-attempt")).To(gomega.Equal([]string{"0"}))
	g.Expect(lastMD.Get("x-throttle-retry-delay")).NotTo(gomega.BeEmpty())
	g.Expect(lastMD.Get("x-retry-delay")).To(gomega.BeEmpty()) // only sent when attempt > 0
}

func TestRetryInterceptorServerDrivenRetryDoesNotCountAgainstClientLimit(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	throttleErr := makeThrottleError(t, 0.01)
	unavailableErr := status.Error(codes.Unavailable, "unavailable")

	callCount := 0
	c := &Client{logger: slog.New(slog.DiscardHandler)}
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		if callCount == 1 {
			return throttleErr
		}
		return unavailableErr
	}

	err := interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).Should(gomega.HaveOccurred())
	// 1 throttle retry + (defaultRetryAttempts+1) client attempts (attempts 0 through defaultRetryAttempts).
	g.Expect(callCount).To(gomega.Equal(1 + defaultRetryAttempts + 1))
}

func TestRetryInterceptorServerDrivenRetryContextCancelled(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	throttleErr := makeThrottleError(t, 60.0) // long delay so the cancel fires first

	ctx, cancel := context.WithCancel(context.Background())
	callCount := 0
	c := &Client{logger: slog.New(slog.DiscardHandler)}
	interceptor := retryInterceptor(c)

	invoker := func(_ context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		cancel()
		return throttleErr
	}

	err := interceptor(ctx, "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(1)) // sleep cancelled immediately, no second attempt
}

func TestRetryInterceptorMaxThrottleWaitDisabled(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	throttleErr := makeThrottleError(t, 0.01)

	callCount := 0
	zero := time.Duration(0)
	c, err := NewClientWithOptions(&ClientParams{Logger: slog.New(slog.DiscardHandler), MaxThrottleWait: &zero})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		return throttleErr
	}

	err = interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	// Server-driven retries are disabled; the throttle error is returned immediately.
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(1))
}

func TestRetryInterceptorMaxThrottleWaitExceeded(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// Server asks for a 60-second delay; cap is 10 seconds → should stop immediately.
	throttleErr := makeThrottleError(t, 60.0)

	callCount := 0
	cap := 10 * time.Second
	c, err := NewClientWithOptions(&ClientParams{Logger: slog.New(slog.DiscardHandler), MaxThrottleWait: &cap})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		return throttleErr
	}

	err = interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(1)) // stopped before sleeping
}

func TestRetryInterceptorMaxThrottleWaitNotExceeded(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// Server asks for a 0.01-second delay; cap is 10 seconds → retries should proceed normally.
	throttleErr := makeThrottleError(t, 0.01)

	callCount := 0
	cap := 10 * time.Second
	c, err := NewClientWithOptions(&ClientParams{Logger: slog.New(slog.DiscardHandler), MaxThrottleWait: &cap})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		if callCount <= 2 {
			return throttleErr
		}
		return nil
	}

	err = interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(3)) // 2 throttle failures + 1 success
}

func TestRetryInterceptorMaxThrottleWaitNil(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// nil means no cap: retries should proceed normally regardless of server delay.
	throttleErr := makeThrottleError(t, 0.01)

	callCount := 0
	c, err := NewClientWithOptions(&ClientParams{Logger: slog.New(slog.DiscardHandler), MaxThrottleWait: nil})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		if callCount <= 2 {
			return throttleErr
		}
		return nil
	}

	err = interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(callCount).To(gomega.Equal(3)) // 2 throttle failures + 1 success
}

func TestRetryInterceptorMaxThrottleWaitConfigHigherPrecedence(t *testing.T) {
	t.Setenv("MODAL_MAX_THROTTLE_WAIT", "30")
	g := gomega.NewWithT(t)

	cap := 10 * time.Second
	c, err := NewClientWithOptions(&ClientParams{Logger: slog.New(slog.DiscardHandler), MaxThrottleWait: &cap})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(*c.profile.MaxThrottleWait).Should(gomega.Equal(cap))
}

func TestRetryInterceptorServerDrivenRetryLogsWarning(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	throttleErr := makeThrottleError(t, 0.01)

	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn}))

	callCount := 0
	c := &Client{logger: logger}
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		if callCount <= 2 {
			return throttleErr
		}
		return nil
	}

	err := interceptor(context.Background(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(buf.String()).To(gomega.ContainSubstring("Server requested retry delay"))
}
