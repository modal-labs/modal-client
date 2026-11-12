package modal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
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

func TestInjectRequiredHeadersWithOAuthCredentials(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	c := &Client{
		profile: Profile{
			OAuthRefreshToken: "refresh-token",
			OAuthClientID:     "oc-client-id",
			OAuthClientSecret: "ov-client-secret",
		},
		sdkVersion: "test-version",
	}

	ctx, err := injectRequiredHeaders(context.Background(), c, "api.modal.com")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	md, ok := metadata.FromOutgoingContext(ctx)
	g.Expect(ok).To(gomega.BeTrue())
	g.Expect(md.Get("x-modal-refresh-token")).To(gomega.Equal([]string{"refresh-token"}))
	g.Expect(md.Get("x-modal-oauth-client-id")).To(gomega.Equal([]string{"oc-client-id"}))
	g.Expect(md.Get("x-modal-oauth-client-secret")).To(gomega.Equal([]string{"ov-client-secret"}))
	g.Expect(md.Get("x-modal-token-id")).To(gomega.BeEmpty())
	g.Expect(md.Get("x-modal-token-secret")).To(gomega.BeEmpty())
}

func TestValidateProfileCredentials(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                        string
		profile                     Profile
		hasExplicitOAuthCredentials bool
		message                     string
	}{
		{
			name: "complete OAuth credentials",
			profile: Profile{
				OAuthRefreshToken: "refresh-token",
				OAuthClientID:     "oc-client-id",
				OAuthClientSecret: "ov-client-secret",
			},
		},
		{
			name:    "incomplete OAuth credentials",
			profile: Profile{OAuthRefreshToken: "refresh-token"},
			message: "exactly one of client secret or JWT key",
		},
		{
			name: "complete OAuth JWT credentials",
			profile: Profile{
				OAuthRefreshToken: "refresh-token",
				OAuthClientID:     "oc-client-id",
				OAuthJWTKey:       "unused-here",
			},
		},
		{
			name: "OAuth secret and JWT key together",
			profile: Profile{
				OAuthRefreshToken: "refresh-token",
				OAuthClientID:     "oc-client-id",
				OAuthClientSecret: "ov-client-secret",
				OAuthJWTKey:       "private-key",
			},
			message: "exactly one of client secret or JWT key",
		},
		{
			name: "empty OAuth refresh token",
			profile: Profile{
				OAuthClientID:     "oc-client-id",
				OAuthClientSecret: "ov-client-secret",
			},
			message: "exactly one of client secret or JWT key",
		},
		{
			name: "mixed credentials",
			profile: Profile{
				TokenID:           "ak-token-id",
				TokenSecret:       "as-token-secret",
				OAuthRefreshToken: "refresh-token",
				OAuthClientID:     "oc-client-id",
				OAuthClientSecret: "ov-client-secret",
			},
			message: "cannot both be configured",
		},
		{
			name: "explicit empty OAuth credentials with token profile",
			profile: Profile{
				TokenID:     "ak-token-id",
				TokenSecret: "as-token-secret",
			},
			hasExplicitOAuthCredentials: true,
			message:                     "cannot both be configured",
		},
		{
			name:                        "explicit empty OAuth credentials",
			hasExplicitOAuthCredentials: true,
			message:                     "exactly one of client secret or JWT key",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			err := validateProfileCredentials(test.profile, test.hasExplicitOAuthCredentials)
			if test.message == "" {
				g.Expect(err).ShouldNot(gomega.HaveOccurred())
			} else {
				g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring(test.message)))
			}
		})
	}
}

func TestExplicitOAuthCredentialsDoNotBackfillFromProfile(t *testing.T) {
	t.Setenv("MODAL_OAUTH_REFRESH_TOKEN", "profile-refresh-token")
	t.Setenv("MODAL_OAUTH_CLIENT_ID", "profile-client-id")
	t.Setenv("MODAL_OAUTH_CLIENT_SECRET", "profile-client-secret")

	for _, test := range []struct {
		name        string
		credentials *OAuthCredentialsParams
	}{
		{name: "all empty", credentials: &OAuthCredentialsParams{}},
		{name: "partial", credentials: &OAuthCredentialsParams{RefreshToken: "explicit-refresh-token"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			g := gomega.NewWithT(t)
			_, err := NewClientWithOptions(&ClientParams{OAuthCredentials: test.credentials})
			g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("exactly one of client secret or JWT key")))
		})
	}
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

func TestHeaderInjectorUnaryInterceptor(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	c := &Client{
		profile:    Profile{TokenID: "test-token-id", TokenSecret: "test-token-secret"},
		sdkVersion: "1.2.3",
	}
	interceptor := headerInjectorUnaryInterceptor(c, "api.modal.com")

	var md metadata.MD
	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		md, _ = metadata.FromOutgoingContext(ctx)
		return nil
	}

	err := interceptor(t.Context(), "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(md.Get("x-modal-host")).To(gomega.Equal([]string{"api.modal.com"}))
	g.Expect(md.Get("x-modal-token-id")).To(gomega.Equal([]string{"test-token-id"}))
	g.Expect(md.Get("x-modal-libmodal-version")).To(gomega.Equal([]string{"modal-go/1.2.3"}))
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

func TestRetryInterceptorThrottleExceedingContextDeadline(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	throttleErr := makeThrottleError(t, 60.0)

	callCount := 0
	c, err := NewClientWithOptions(&ClientParams{Logger: slog.New(slog.DiscardHandler)})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	interceptor := retryInterceptor(c)

	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		callCount++
		return throttleErr
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	err = interceptor(ctx, "/modal.client.ModalClient/AppGetOrCreate", nil, nil, nil, invoker)
	g.Expect(err).To(gomega.MatchError(throttleErr))
	g.Expect(callCount).To(gomega.Equal(1))
	g.Expect(time.Since(started)).To(gomega.BeNumerically("<", 40*time.Millisecond))
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

// fakeServer records the requests a server started by startFakeServer receives.
type fakeServer struct {
	hits atomic.Int32
	// hosts holds the x-modal-host values of the most recent request.
	hosts atomic.Value
}

// startFakeServer serves gRPC on a local port and records the requests it receives. Every request
// fails with PermissionDenied, which the client does not retry.
func startFakeServer(t *testing.T) (string, *fakeServer) {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	gomega.NewWithT(t).Expect(err).ToNot(gomega.HaveOccurred())
	fake := &fakeServer{}
	srv := grpc.NewServer(grpc.UnknownServiceHandler(func(_ any, stream grpc.ServerStream) error {
		fake.hits.Add(1)
		md, _ := metadata.FromIncomingContext(stream.Context())
		fake.hosts.Store(md.Get("x-modal-host"))
		return status.Error(codes.PermissionDenied, "fake server")
	}))
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return "http://" + lis.Addr().String(), fake
}

// closedPortURL returns a URL on a local port with no listener, so connections are refused.
func closedPortURL(t *testing.T) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	gomega.NewWithT(t).Expect(err).ToNot(gomega.HaveOccurred())
	addr := lis.Addr().String()
	gomega.NewWithT(t).Expect(lis.Close()).To(gomega.Succeed())
	return "http://" + addr
}

// blackholeURL returns a URL on a local port that accepts TCP connections but never responds.
func blackholeURL(t *testing.T) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	gomega.NewWithT(t).Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = lis.Close() })
	go func() {
		for {
			conn, err := lis.Accept()
			if err != nil {
				return
			}
			t.Cleanup(func() { _ = conn.Close() })
		}
	}()
	return "http://" + lis.Addr().String()
}

// newClientForServerURL creates a client for serverURL and makes one request through it, so the
// caller can check which fake server received the request.
func newClientForServerURL(t *testing.T, serverURL string) {
	t.Helper()
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_SERVER_URL", serverURL)
	c, err := NewClientWithOptions(&ClientParams{TokenID: "ak-test", TokenSecret: "as-test", Logger: slog.New(slog.DiscardHandler)})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(c.Close)
	_, err = c.Secrets.FromName(context.Background(), "my-secret", nil)
	g.Expect(status.Code(errors.Unwrap(err))).To(gomega.Equal(codes.PermissionDenied))
}

func TestServerURLFallbackWhenPrimaryUnreachable(t *testing.T) {
	for name, primary := range map[string]func(*testing.T) string{
		"connection refused": closedPortURL,
		"dns failure":        func(*testing.T) string { return "http://modal-test-host.invalid:443" },
		"tls to plaintext": func(t *testing.T) string {
			url, _ := startFakeServer(t)
			return "https://" + url[len("http://"):]
		},
		"blackholed": blackholeURL,
	} {
		t.Run(name, func(t *testing.T) {
			g := gomega.NewWithT(t)
			secondary, fake := startFakeServer(t)
			newClientForServerURL(t, primary(t)+","+secondary)
			g.Expect(fake.hits.Load()).To(gomega.BeNumerically(">", 0))
		})
	}
}

func TestServerURLFallbackFailsOverWithoutStaggerOnFastFailure(t *testing.T) {
	g := gomega.NewWithT(t)
	secondary, fake := startFakeServer(t)
	start := time.Now()
	newClientForServerURL(t, closedPortURL(t)+","+secondary)
	g.Expect(time.Since(start)).To(gomega.BeNumerically("<", fallbackStaggerDelay))
	g.Expect(fake.hits.Load()).To(gomega.BeNumerically(">", 0))
}

func TestServerURLFallbackWaitsOutStaggerOnSilentPrimary(t *testing.T) {
	g := gomega.NewWithT(t)
	secondary, fake := startFakeServer(t)
	start := time.Now()
	newClientForServerURL(t, blackholeURL(t)+","+secondary)
	g.Expect(time.Since(start)).To(gomega.BeNumerically(">=", fallbackStaggerDelay))
	g.Expect(fake.hits.Load()).To(gomega.BeNumerically(">", 0))
}

func TestServerURLFallbackPrefersPrimaryWhenReachable(t *testing.T) {
	g := gomega.NewWithT(t)
	primary, primaryFake := startFakeServer(t)
	secondary, secondaryFake := startFakeServer(t)
	// Whitespace around the separator is ignored.
	newClientForServerURL(t, primary+" , "+secondary)
	g.Expect(primaryFake.hits.Load()).To(gomega.BeNumerically(">", 0))
	g.Expect(secondaryFake.hits.Load()).To(gomega.BeZero())
}

// localhostURL returns url, a URL on 127.0.0.1, with its host spelled "localhost". It reaches the
// same server under a different host name. Tests only use it for a secondary URL: localhost may
// resolve to ::1 first, and that failed attempt must not let a secondary race a reachable primary.
func localhostURL(url string) string {
	return strings.Replace(url, "127.0.0.1", "localhost", 1)
}

func TestServerURLFallbackSendsPrimaryHostWhenPrimaryReachable(t *testing.T) {
	g := gomega.NewWithT(t)
	primary, primaryFake := startFakeServer(t)
	secondary, _ := startFakeServer(t)
	newClientForServerURL(t, primary+","+localhostURL(secondary))
	g.Expect(primaryFake.hosts.Load()).To(gomega.HaveEach("127.0.0.1"))
}

func TestServerURLFallbackSendsSecondaryHostWhenPrimaryUnreachable(t *testing.T) {
	g := gomega.NewWithT(t)
	secondary, secondaryFake := startFakeServer(t)
	newClientForServerURL(t, closedPortURL(t)+","+localhostURL(secondary))
	g.Expect(secondaryFake.hosts.Load()).To(gomega.HaveEach("localhost"))
}

func TestServerURLFallbackUsesThirdURLWhenFirstTwoFail(t *testing.T) {
	g := gomega.NewWithT(t)
	third, fake := startFakeServer(t)
	newClientForServerURL(t, closedPortURL(t)+","+closedPortURL(t)+","+third)
	g.Expect(fake.hits.Load()).To(gomega.BeNumerically(">", 0))
}

func TestSingleServerURLConnectsLazily(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_SERVER_URL", closedPortURL(t))
	start := time.Now()
	c, err := NewClientWithOptions(&ClientParams{TokenID: "ak-test", TokenSecret: "as-test", Logger: slog.New(slog.DiscardHandler)})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer c.Close()
	g.Expect(time.Since(start)).To(gomega.BeNumerically("<", time.Second))
}

func TestServerURLFallbackErrorsWhenAllURLsUnreachable(t *testing.T) {
	g := gomega.NewWithT(t)
	// NewClientWithOptions takes no context, so call the dialer directly to bound the wait.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	c := &Client{logger: slog.New(slog.DiscardHandler)}
	serverURL := closedPortURL(t) + "," + closedPortURL(t)
	conn, _, err := dialWithFallbacks(ctx, Profile{ServerURL: serverURL}, c)
	g.Expect(conn).To(gomega.BeNil())
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(serverURL)))
}
