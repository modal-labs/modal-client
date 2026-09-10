package modal

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// newStubClient builds a client through the constructor and swaps its stub for
// a fake, so a fixture carries whatever fields a client needs without listing
// them.
func newStubClient(t *testing.T, stub pb.TaskCommandRouterClient) *taskCommandRouterClient {
	t.Helper()
	client, err := newTaskCommandRouterClient(commandRouterParams{
		taskID: "ta-1",
		jwt:    mockJWT(time.Now().Unix() + 3600),
		// Never reached: the stub below stands in for the connection.
		target: "passthrough:///unused",
		creds:  insecure.NewCredentials(),
		logger: slog.New(slog.DiscardHandler),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	client.stubValue = stub
	return client
}

type drainingRouterStub struct {
	pb.TaskCommandRouterClient
	start func(context.Context) (*pb.TaskExecStartResponse, error)
	read  func(context.Context) (grpc.ServerStreamingClient[pb.SandboxStdioReadV2Response], error)
}

func (s *drainingRouterStub) TaskExecStart(ctx context.Context, _ *pb.TaskExecStartRequest, _ ...grpc.CallOption) (*pb.TaskExecStartResponse, error) {
	return s.start(ctx)
}

func (s *drainingRouterStub) SandboxStdioReadV2(ctx context.Context, _ *pb.SandboxStdioReadV2Request, _ ...grpc.CallOption) (grpc.ServerStreamingClient[pb.SandboxStdioReadV2Response], error) {
	return s.read(ctx)
}

type drainingStdioStream struct {
	grpc.ServerStreamingClient[pb.SandboxStdioReadV2Response]
	recv func() (*pb.SandboxStdioReadV2Response, error)
}

func (s *drainingStdioStream) Recv() (*pb.SandboxStdioReadV2Response, error) { return s.recv() }

func TestSandboxDetachDrainsConcurrentOperations(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	started := make(chan int, 2)
	finish := []chan struct{}{make(chan struct{}), make(chan struct{})}
	var calls atomic.Int64
	stub := &drainingRouterStub{start: func(ctx context.Context) (*pb.TaskExecStartResponse, error) {
		index := int(calls.Add(1) - 1)
		started <- index
		select {
		case <-finish[index]:
			return &pb.TaskExecStartResponse{}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}}
	client := newStubClient(t, stub)
	sb := newSandbox(&Client{}, testV2SandboxID)
	sb.commandRouterClient = client
	results := make(chan error, 2)
	for range 2 {
		go func() { _, err := client.ExecStart(t.Context(), &pb.TaskExecStartRequest{}); results <- err }()
	}
	g.Eventually(started).Should(gomega.Receive())
	g.Eventually(started).Should(gomega.Receive())
	g.Expect(sb.Detach()).To(gomega.Succeed())
	g.Expect(client.closed.Load()).To(gomega.BeFalse())
	_, err := client.ExecStart(t.Context(), &pb.TaskExecStartRequest{})
	var closed ClientClosedError
	g.Expect(errors.As(err, &closed)).To(gomega.BeTrue())
	close(finish[0])
	g.Eventually(results).Should(gomega.Receive(gomega.BeNil()))
	g.Expect(client.closed.Load()).To(gomega.BeFalse())
	close(finish[1])
	g.Eventually(results).Should(gomega.Receive(gomega.BeNil()))
	g.Expect(client.closed.Load()).To(gomega.BeTrue())
	client.connMu.Lock()
	g.Expect(client.conn).To(gomega.BeNil())
	client.connMu.Unlock()
}

func TestDetachKeepsUnaryLeaseAcrossRetries(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	stub := &drainingRouterStub{}
	client := newStubClient(t, stub)
	calls := 0
	stub.start = func(context.Context) (*pb.TaskExecStartResponse, error) {
		calls++
		g.Expect(client.closed.Load()).To(gomega.BeFalse())
		if calls == 1 {
			client.closeWhenIdle()
			return nil, status.Error(codes.Unavailable, "retry this attempt")
		}
		return &pb.TaskExecStartResponse{}, nil
	}
	_, err := client.ExecStart(t.Context(), &pb.TaskExecStartRequest{})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(calls).To(gomega.Equal(2))
	g.Expect(client.closed.Load()).To(gomega.BeTrue())
}

func TestDetachKeepsStdioLeaseAcrossRetries(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	stub := &drainingRouterStub{}
	client := newStubClient(t, stub)
	calls := 0
	stub.read = func(context.Context) (grpc.ServerStreamingClient[pb.SandboxStdioReadV2Response], error) {
		calls++
		return &drainingStdioStream{recv: func() (*pb.SandboxStdioReadV2Response, error) {
			g.Expect(client.closed.Load()).To(gomega.BeFalse())
			if calls == 1 {
				client.closeWhenIdle()
				return nil, status.Error(codes.Unavailable, "retry this read")
			}
			return pb.SandboxStdioReadV2Response_builder{Data: []byte("a")}.Build(), nil
		}}, nil
	}
	reader := client.SandboxStdioReadV2(t.Context(), "ta-1", pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT)
	t.Cleanup(func() { _ = reader.Close() })
	buf := make([]byte, 1)
	n, err := reader.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf[:n])).To(gomega.Equal("a"))
	g.Expect(calls).To(gomega.Equal(2))
	g.Expect(client.closed.Load()).To(gomega.BeTrue())
}

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

func TestCallWithRetriesOnTransientErrorsExcludeCodes(t *testing.T) {
	t.Parallel()

	excluded := []codes.Code{codes.DeadlineExceeded, codes.Canceled}
	for _, code := range excluded {
		t.Run(code.String(), func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			ctx := t.Context()
			callCount := 0
			_, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
				callCount++
				return nil, status.Error(code, "")
			}, retryOptions{
				BaseDelay:    time.Millisecond,
				DelayFactor:  1,
				MaxRetries:   intPtr(10),
				ExcludeCodes: excluded,
			}, nil)

			g.Expect(err).To(gomega.HaveOccurred())
			// Excluded codes are not retried, even if they're in the
			// general retryable set.
			g.Expect(callCount).To(gomega.Equal(1))
		})
	}
}

func TestCallWithRetriesOnTransientErrorsExcludeCodesDoesNotAffectOtherRetryableCodes(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	callCount := 0
	result, err := callWithRetriesOnTransientErrors(ctx, func() (*string, error) {
		callCount++
		if callCount == 1 {
			return nil, status.Error(codes.Unavailable, "unavailable")
		}
		out := "ok"
		return &out, nil
	}, retryOptions{
		BaseDelay:    time.Millisecond,
		DelayFactor:  1,
		MaxRetries:   intPtr(10),
		ExcludeCodes: []codes.Code{codes.DeadlineExceeded, codes.Canceled},
	}, nil)

	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(*result).To(gomega.Equal("ok"))
	g.Expect(callCount).To(gomega.Equal(2))
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

// mockSnapshotFsStub embeds pb.TaskCommandRouterClient so unused methods
// inherit nil stubs (calling them would panic). Only TaskSnapshotFilesystem
// is overridden.
type mockSnapshotFsStub struct {
	pb.TaskCommandRouterClient
	fn func(ctx context.Context, in *pb.TaskSnapshotFilesystemRequest, opts ...grpc.CallOption) (*pb.TaskSnapshotFilesystemResponse, error)
}

func (m *mockSnapshotFsStub) TaskSnapshotFilesystem(
	ctx context.Context,
	in *pb.TaskSnapshotFilesystemRequest,
	opts ...grpc.CallOption,
) (*pb.TaskSnapshotFilesystemResponse, error) {
	return m.fn(ctx, in, opts...)
}

// Regression test for a preemptive-deadline error-translation bug.
//
// `callWithRetriesOnTransientErrors` returns `errDeadlineExceeded` as
// soon as the *next* backoff sleep would overshoot the deadline — at
// that moment `time.Now()` is still strictly before the deadline.
// SnapshotFilesystem's outer error translation only converts to
// TimeoutError when `time.Now().After(overallDeadline)`, so the raw
// sentinel leaks through to the caller instead of TimeoutError.
func TestSnapshotFilesystemPreemptiveDeadlineReturnsTimeoutError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	stub := &mockSnapshotFsStub{
		fn: func(_ context.Context, _ *pb.TaskSnapshotFilesystemRequest, _ ...grpc.CallOption) (*pb.TaskSnapshotFilesystemResponse, error) {
			return nil, status.Error(codes.Unavailable, "transient")
		},
	}

	client := newStubClient(t, stub)
	jwt := "fake-jwt"
	client.jwt.Store(&jwt)

	// With BaseDelay=10ms doubling each retry, a 100ms timeout will
	// produce attempts at t≈0,10,30,70 — and the 5th retry's 80ms wait
	// pushes past the 100ms deadline before time.Now() crosses it.
	_, err := client.SnapshotFilesystem(
		context.Background(),
		&pb.TaskSnapshotFilesystemRequest{},
		100*time.Millisecond,
	)
	g.Expect(err).To(gomega.HaveOccurred())

	_, isTimeout := err.(TimeoutError)
	g.Expect(isTimeout).To(gomega.BeTrue(),
		"expected TimeoutError, got %T: %v", err, err)
}

// mockWaitUntilReadyClientStub embeds pb.TaskCommandRouterClient so unused
// methods inherit nil stubs. Only SandboxWaitUntilReady is overridden.
type mockWaitUntilReadyClientStub struct {
	pb.TaskCommandRouterClient
	fn func(ctx context.Context, in *pb.SandboxWaitUntilReadyTcrRequest, opts ...grpc.CallOption) (*pb.SandboxWaitUntilReadyTcrResponse, error)
}

func (m *mockWaitUntilReadyClientStub) SandboxWaitUntilReady(
	ctx context.Context,
	in *pb.SandboxWaitUntilReadyTcrRequest,
	opts ...grpc.CallOption,
) (*pb.SandboxWaitUntilReadyTcrResponse, error) {
	return m.fn(ctx, in, opts...)
}

func TestSandboxWaitUntilReadyReturnsTimeoutErrorOnDeadline(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	calls := 0
	stub := &mockWaitUntilReadyClientStub{
		fn: func(_ context.Context, _ *pb.SandboxWaitUntilReadyTcrRequest, _ ...grpc.CallOption) (*pb.SandboxWaitUntilReadyTcrResponse, error) {
			calls++
			return nil, status.Error(codes.DeadlineExceeded, "deadline exceeded")
		},
	}

	client := newStubClient(t, stub)
	jwt := "fake-jwt"
	client.jwt.Store(&jwt)

	_, err := client.SandboxWaitUntilReady(context.Background(), "ta-123", 100*time.Millisecond)
	var timeoutErr TimeoutError
	g.Expect(errors.As(err, &timeoutErr)).To(gomega.BeTrue(),
		"expected TimeoutError, got %T: %v", err, err)
}

// fakeStdinRouterServer is a minimal in-process TaskCommandRouter
// implementation covering just the streaming-stdin RPCs, with configurable
// fault injection to exercise the resume logic in ExecStdinWriteStream.
type fakeStdinRouterServer struct {
	pb.UnimplementedTaskCommandRouterServer

	mu           sync.Mutex
	received     []byte
	closed       bool
	streamStarts int
	statusCalls  int

	// failuresRemaining injects a stream abort on a Data message while > 0.
	failuresRemaining int
	// failAfterBytes only injects a failure once total received bytes
	// (including the triggering chunk) reach this threshold.
	failAfterBytes int
	// recordBytesOnFailure controls whether the triggering chunk is recorded
	// before the injected failure, i.e. whether the failed attempt made progress.
	recordBytesOnFailure bool
	// failCode is the injected status code; defaults to Unavailable.
	failCode codes.Code
	// failOnEnd closes stdin on End but aborts the stream instead of sending
	// the response, simulating a lost response after a completed upload.
	failOnEnd bool
	// succeedEarlyAfterBytes, when > 0, completes the RPC successfully once
	// total received bytes reach the threshold, without waiting for End. The
	// client's in-flight Send then observes io.EOF for a successful RPC.
	succeedEarlyAfterBytes int
}

func (s *fakeStdinRouterServer) injectedCode() codes.Code {
	if s.failCode == codes.OK {
		return codes.Unavailable
	}
	return s.failCode
}

func (s *fakeStdinRouterServer) TaskExecStdinWriteStream(
	stream grpc.ClientStreamingServer[pb.TaskExecStdinWriteStreamRequest, pb.TaskExecStdinWriteStreamResponse],
) error {
	s.mu.Lock()
	s.streamStarts++
	s.mu.Unlock()

	first, err := stream.Recv()
	if err != nil {
		return err
	}
	start := first.GetStart()
	if start == nil {
		return status.Error(codes.InvalidArgument, "first message must be start")
	}
	s.mu.Lock()
	if start.GetOffset() > uint64(len(s.received)) {
		s.mu.Unlock()
		return status.Error(codes.InvalidArgument, "offset beyond received bytes")
	}
	s.received = s.received[:start.GetOffset()]
	s.mu.Unlock()

	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			// The client went away without an explicit End: leave stdin open.
			return nil
		}
		if err != nil {
			return err
		}
		switch msg.WhichPayload() {
		case pb.TaskExecStdinWriteStreamRequest_Data_case:
			s.mu.Lock()
			inject := s.failuresRemaining > 0 && len(s.received)+len(msg.GetData()) >= s.failAfterBytes
			if inject {
				s.failuresRemaining--
			}
			if !inject || s.recordBytesOnFailure {
				s.received = append(s.received, msg.GetData()...)
			}
			code := s.injectedCode()
			succeedEarly := s.succeedEarlyAfterBytes > 0 && len(s.received) >= s.succeedEarlyAfterBytes
			if succeedEarly {
				s.closed = true
			}
			s.mu.Unlock()
			if inject {
				return status.Error(code, "injected stream failure")
			}
			if succeedEarly {
				return stream.SendAndClose(&pb.TaskExecStdinWriteStreamResponse{})
			}
		case pb.TaskExecStdinWriteStreamRequest_End_case:
			s.mu.Lock()
			s.closed = true
			failOnEnd := s.failOnEnd
			s.mu.Unlock()
			if failOnEnd {
				return status.Error(codes.Unavailable, "injected failure after End")
			}
			return stream.SendAndClose(&pb.TaskExecStdinWriteStreamResponse{})
		default:
			return status.Error(codes.InvalidArgument, "unexpected payload")
		}
	}
}

func (s *fakeStdinRouterServer) TaskExecStdinStatus(
	ctx context.Context,
	req *pb.TaskExecStdinStatusRequest,
) (*pb.TaskExecStdinStatusResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.statusCalls++
	return pb.TaskExecStdinStatusResponse_builder{
		NumBytesWritten: uint64(len(s.received)),
		Closed:          s.closed,
	}.Build(), nil
}

func (s *fakeStdinRouterServer) snapshot() (received []byte, closed bool, streamStarts, statusCalls int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]byte(nil), s.received...), s.closed, s.streamStarts, s.statusCalls
}

// newStreamingStdinTestClient serves fake over an in-process gRPC server and
// returns a taskCommandRouterClient connected to it.
func newStreamingStdinTestClient(t *testing.T, fake *fakeStdinRouterServer) *taskCommandRouterClient {
	t.Helper()
	g := gomega.NewWithT(t)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	grpcServer := grpc.NewServer()
	pb.RegisterTaskCommandRouterServer(grpcServer, fake)
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	t.Cleanup(grpcServer.Stop)

	client, err := newTaskCommandRouterClient(commandRouterParams{
		taskID: "ta-1",
		jwt:    mockJWT(time.Now().Unix() + 3600),
		target: lis.Addr().String(),
		creds:  insecure.NewCredentials(),
		logger: slog.New(slog.DiscardHandler),
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// deterministicBytes returns size bytes with a pattern that catches
// chunk-boundary mistakes (unlike a constant fill).
func deterministicBytes(size int) []byte {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i*31 + i/streamingStdinChunkSize)
	}
	return data
}

func TestExecStdinWriteStreamHappyPathMultiChunk(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdinRouterServer{}
	client := newStreamingStdinTestClient(t, fake)

	// Two full chunks plus a partial third.
	payload := deterministicBytes(2*streamingStdinChunkSize + 100)
	n, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(payload))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(int64(len(payload))))

	received, closed, streamStarts, statusCalls := fake.snapshot()
	g.Expect(received).To(gomega.Equal(payload))
	g.Expect(closed).To(gomega.BeTrue())
	g.Expect(streamStarts).To(gomega.Equal(1))
	g.Expect(statusCalls).To(gomega.Equal(0))
}

func TestExecStdinWriteStreamSucceedsWhenServerCompletesEarly(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// A server that completes the RPC before consuming the whole stream makes
	// the client's in-flight Send return io.EOF; the successful CloseAndRecv
	// must be treated as success rather than surfacing io.EOF as an error.
	fake := &fakeStdinRouterServer{succeedEarlyAfterBytes: streamingStdinChunkSize}
	client := newStreamingStdinTestClient(t, fake)

	payload := deterministicBytes(16 * streamingStdinChunkSize)
	_, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(payload))
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, _, streamStarts, statusCalls := fake.snapshot()
	g.Expect(streamStarts).To(gomega.Equal(1))
	g.Expect(statusCalls).To(gomega.Equal(0))
}

func TestExecStdinWriteStreamEmptySourceClosesStdin(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdinRouterServer{}
	client := newStreamingStdinTestClient(t, fake)

	n, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(nil))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(int64(0)))

	received, closed, _, _ := fake.snapshot()
	g.Expect(received).To(gomega.BeEmpty())
	g.Expect(closed).To(gomega.BeTrue())
}

func TestExecStdinWriteStreamResumesFromReportedOffset(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdinRouterServer{
		failuresRemaining:    1,
		failAfterBytes:       streamingStdinChunkSize + 1,
		recordBytesOnFailure: true,
	}
	client := newStreamingStdinTestClient(t, fake)

	payload := deterministicBytes(3*streamingStdinChunkSize + 100)
	n, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(payload))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(int64(len(payload))))

	received, closed, streamStarts, statusCalls := fake.snapshot()
	g.Expect(received).To(gomega.Equal(payload))
	g.Expect(closed).To(gomega.BeTrue())
	g.Expect(streamStarts).To(gomega.Equal(2))
	g.Expect(statusCalls).To(gomega.Equal(1))
}

func TestExecStdinWriteStreamExhaustsResumeAttempts(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// Every attempt fails on its first Data chunk without making progress.
	fake := &fakeStdinRouterServer{
		failuresRemaining:    1000,
		recordBytesOnFailure: false,
	}
	client := newStreamingStdinTestClient(t, fake)

	_, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(deterministicBytes(100)))
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(status.Code(err)).To(gomega.Equal(codes.Unavailable))

	_, closed, streamStarts, statusCalls := fake.snapshot()
	g.Expect(closed).To(gomega.BeFalse())
	// 10 total attempts (initial plus 9 resumes), matching the unary retry budget.
	g.Expect(streamStarts).To(gomega.Equal(10))
	g.Expect(statusCalls).To(gomega.Equal(9))
}

func TestExecStdinWriteStreamRecoversWhenResponseLostAfterEnd(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdinRouterServer{failOnEnd: true}
	client := newStreamingStdinTestClient(t, fake)

	payload := deterministicBytes(streamingStdinChunkSize + 100)
	n, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(payload))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(int64(len(payload))))

	received, closed, streamStarts, statusCalls := fake.snapshot()
	g.Expect(received).To(gomega.Equal(payload))
	g.Expect(closed).To(gomega.BeTrue())
	g.Expect(streamStarts).To(gomega.Equal(1))
	g.Expect(statusCalls).To(gomega.Equal(1))
}

func TestExecStdinWriteStreamDoesNotResumeOnFailedPrecondition(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdinRouterServer{
		failuresRemaining: 1,
		failCode:          codes.FailedPrecondition,
	}
	client := newStreamingStdinTestClient(t, fake)

	_, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", bytes.NewReader(deterministicBytes(100)))
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(status.Code(err)).To(gomega.Equal(codes.FailedPrecondition))

	_, _, streamStarts, statusCalls := fake.snapshot()
	g.Expect(streamStarts).To(gomega.Equal(1))
	g.Expect(statusCalls).To(gomega.Equal(0))
}

// failingReadSeeker fails reads after a prefix, simulating a local source error.
type failingReadSeeker struct {
	*bytes.Reader
	failAt int64
	err    error
}

type detachingReadSeeker struct {
	*bytes.Reader
	once   sync.Once
	client *taskCommandRouterClient
}

func (s *detachingReadSeeker) Read(p []byte) (int, error) {
	s.once.Do(s.client.closeWhenIdle)
	return s.Reader.Read(p)
}

func TestDetachKeepsStdinLeaseAcrossResumeAndStatusCalls(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	fake := &fakeStdinRouterServer{failuresRemaining: 1, failCode: codes.Unavailable, failAfterBytes: 1}
	client := newStreamingStdinTestClient(t, fake)
	data := deterministicBytes(100)
	source := &detachingReadSeeker{Reader: bytes.NewReader(data), client: client}
	_, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", source)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	received, closed, starts, statusCalls := fake.snapshot()
	g.Expect(received).To(gomega.Equal(data))
	g.Expect(closed).To(gomega.BeTrue())
	g.Expect(starts).To(gomega.Equal(2))
	g.Expect(statusCalls).To(gomega.Equal(1))
	g.Expect(client.closed.Load()).To(gomega.BeTrue())
}

func (f *failingReadSeeker) Read(p []byte) (int, error) {
	pos, seekErr := f.Seek(0, io.SeekCurrent)
	if seekErr != nil {
		return 0, seekErr
	}
	if pos >= f.failAt {
		return 0, f.err
	}
	if remaining := f.failAt - pos; int64(len(p)) > remaining {
		p = p[:remaining]
	}
	return f.Reader.Read(p)
}

func TestExecStdinWriteStreamLocalReadErrorIsNotResumed(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdinRouterServer{}
	client := newStreamingStdinTestClient(t, fake)

	readErr := errors.New("local disk exploded")
	source := &failingReadSeeker{
		Reader: bytes.NewReader(deterministicBytes(2 * streamingStdinChunkSize)),
		failAt: 10,
		err:    readErr,
	}
	_, err := client.ExecStdinWriteStream(t.Context(), "ta-1", "ex-1", source)
	g.Expect(err).To(gomega.MatchError(readErr))

	// The server handler finishes asynchronously after the client abandons the
	// stream, so poll for the stream to have been observed.
	g.Eventually(func() int {
		_, _, streamStarts, _ := fake.snapshot()
		return streamStarts
	}).Should(gomega.Equal(1))
	_, closed, _, statusCalls := fake.snapshot()
	// The failed attempt must not send End: stdin stays open server-side.
	g.Expect(closed).To(gomega.BeFalse())
	g.Expect(statusCalls).To(gomega.Equal(0))
}

type fakeStdioRouterServer struct {
	pb.UnimplementedTaskCommandRouterServer

	mu         sync.Mutex
	stdoutData []byte
	stderrData []byte

	chunkSize         int
	droppedPrefix     int
	failuresRemaining int
	failCode          codes.Code

	readCalls     int
	stdinReceived []byte
	stdinClosed   bool
	stdinCalls    int
}

type guardedStdioStream[T any] struct {
	grpc.ClientStream
	response  *T
	recvCalls int
}

func (s *guardedStdioStream[T]) Recv() (*T, error) {
	s.recvCalls++
	if s.response != nil {
		response := s.response
		s.response = nil
		return response, nil
	}
	return nil, status.Error(codes.Unavailable, "stale stream was polled")
}

type eofStdioStream[T any] struct {
	grpc.ClientStream
	recvCalls int
}

func (s *eofStdioStream[T]) Recv() (*T, error) {
	s.recvCalls++
	return nil, io.EOF
}

type releasedStdioClient struct {
	pb.TaskCommandRouterClient
	sandboxStale       *guardedStdioStream[pb.SandboxStdioReadV2Response]
	sandboxReplacement *eofStdioStream[pb.SandboxStdioReadV2Response]
	execStale          *guardedStdioStream[pb.TaskExecStdioReadResponse]
	execReplacement    *eofStdioStream[pb.TaskExecStdioReadResponse]
	offsets            []uint64
}

func (c *releasedStdioClient) SandboxStdioReadV2(
	_ context.Context,
	req *pb.SandboxStdioReadV2Request,
	_ ...grpc.CallOption,
) (grpc.ServerStreamingClient[pb.SandboxStdioReadV2Response], error) {
	c.offsets = append(c.offsets, req.GetOffset())
	if req.GetOffset() == 0 {
		return c.sandboxStale, nil
	}
	return c.sandboxReplacement, nil
}

func (c *releasedStdioClient) TaskExecStdioRead(
	_ context.Context,
	req *pb.TaskExecStdioReadRequest,
	_ ...grpc.CallOption,
) (grpc.ServerStreamingClient[pb.TaskExecStdioReadResponse], error) {
	c.offsets = append(c.offsets, req.GetOffset())
	if req.GetOffset() == 0 {
		return c.execStale, nil
	}
	return c.execReplacement, nil
}

func (s *fakeStdioRouterServer) SandboxStdioReadV2(
	req *pb.SandboxStdioReadV2Request,
	stream grpc.ServerStreamingServer[pb.SandboxStdioReadV2Response],
) error {
	s.mu.Lock()
	s.readCalls++
	var full []byte
	switch req.GetFileDescriptor() {
	case pb.SandboxStdioFileDescriptor_SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT:
		full = s.stdoutData
	case pb.SandboxStdioFileDescriptor_SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR:
		full = s.stderrData
	default:
		s.mu.Unlock()
		return status.Errorf(codes.InvalidArgument, "unexpected file descriptor %v", req.GetFileDescriptor())
	}
	fail := s.failuresRemaining > 0
	if fail {
		s.failuresRemaining--
	}
	chunkSize := s.chunkSize
	dropped := s.droppedPrefix
	failCode := s.failCode
	s.mu.Unlock()

	if failCode == codes.OK {
		failCode = codes.Unavailable
	}
	if chunkSize <= 0 {
		chunkSize = len(full)
	}

	off := int(req.GetOffset())
	if off < dropped {
		off = dropped
	}
	for off < len(full) {
		end := off + chunkSize
		if end > len(full) {
			end = len(full)
		}
		resp := pb.SandboxStdioReadV2Response_builder{
			Data:           append([]byte(nil), full[off:end]...),
			StartingOffset: uint64(off),
		}.Build()
		if err := stream.Send(resp); err != nil {
			return err
		}
		off = end
		if fail {
			return status.Error(failCode, "injected read failure")
		}
	}
	return nil
}

func (s *fakeStdioRouterServer) SandboxStdinWriteV2(
	_ context.Context,
	req *pb.SandboxStdinWriteV2Request,
) (*pb.SandboxStdinWriteV2Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stdinCalls++
	if req.GetOffset() > uint64(len(s.stdinReceived)) {
		return nil, status.Error(codes.InvalidArgument, "offset beyond received bytes")
	}
	s.stdinReceived = append(s.stdinReceived[:req.GetOffset()], req.GetData()...)
	if req.GetEof() {
		s.stdinClosed = true
	}
	return &pb.SandboxStdinWriteV2Response{}, nil
}

func (s *fakeStdioRouterServer) readCallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readCalls
}

func (s *fakeStdioRouterServer) stdinSnapshot() (received []byte, closed bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]byte(nil), s.stdinReceived...), s.stdinClosed
}

func newStdioTestClient(t *testing.T, fake *fakeStdioRouterServer) *taskCommandRouterClient {
	t.Helper()
	g := gomega.NewWithT(t)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	grpcServer := grpc.NewServer()
	pb.RegisterTaskCommandRouterServer(grpcServer, fake)
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	t.Cleanup(grpcServer.Stop)

	client, err := newTaskCommandRouterClient(commandRouterParams{
		taskID:    "ta-1",
		sandboxID: testV2SandboxID,
		jwt:       mockJWT(time.Now().Unix() + 3600),
		target:    lis.Addr().String(),
		creds:     insecure.NewCredentials(),
		logger:    slog.New(slog.DiscardHandler),
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func TestSandboxStdioReadV2HappyPath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdioRouterServer{stdoutData: deterministicBytes(5000), chunkSize: 1000}
	client := newStdioTestClient(t, fake)

	reader := client.SandboxStdioReadV2(t.Context(), "ta-1", pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT)
	defer func() { _ = reader.Close() }()

	g.Expect(fake.readCallCount()).To(gomega.Equal(0),
		"no read has happened, so the worker should not have been asked for output")

	got, err := io.ReadAll(reader)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal(fake.stdoutData))
	g.Expect(fake.readCallCount()).To(gomega.Equal(1))
}

func TestSandboxStdioReadV2ResumesAfterTransientError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// Abort after the first chunk, then resume on reconnect from the client's offset.
	fake := &fakeStdioRouterServer{stdoutData: deterministicBytes(5000), chunkSize: 1000, failuresRemaining: 1}
	client := newStdioTestClient(t, fake)

	reader := client.SandboxStdioReadV2(t.Context(), "ta-1", pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT)
	defer func() { _ = reader.Close() }()

	got, err := io.ReadAll(reader)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal(fake.stdoutData))
	g.Expect(fake.readCallCount()).To(gomega.Equal(2))
}

func assertStdioReaderReopensReleasedStreamWithoutRetry(
	t *testing.T,
	client *taskCommandRouterClient,
	reader io.ReadCloser,
	retry *stdioRetry,
	staleRecvCalls *int,
	replacementRecvCalls *int,
	offsets *[]uint64,
) {
	t.Helper()
	g := gomega.NewWithT(t)
	defer func() { _ = reader.Close() }()

	buf := make([]byte, 1)
	n, err := reader.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(buf[:n]).To(gomega.Equal([]byte("a")))

	retry.retriesRemaining = 3
	retry.delay = 123 * time.Millisecond
	// Model the reconnect performed after the idle countdown releases the old
	// connection. The next read must replace its stream before touching it.
	client.connMu.Lock()
	client.generation++
	client.connMu.Unlock()

	n, err = reader.Read(buf)
	g.Expect(err).To(gomega.Equal(io.EOF))
	g.Expect(n).To(gomega.Equal(0))
	g.Expect(*staleRecvCalls).To(gomega.Equal(1), "the released stream must not be polled again")
	g.Expect(*replacementRecvCalls).To(gomega.Equal(1))
	g.Expect(*offsets).To(gomega.Equal([]uint64{0, 1}))
	g.Expect(retry.retriesRemaining).To(gomega.Equal(3))
	g.Expect(retry.delay).To(gomega.Equal(123 * time.Millisecond))
}

func TestSandboxStdioReadV2ReopensReleasedStreamWithoutRetry(t *testing.T) {
	t.Parallel()

	stale := &guardedStdioStream[pb.SandboxStdioReadV2Response]{
		response: pb.SandboxStdioReadV2Response_builder{Data: []byte("a")}.Build(),
	}
	replacement := &eofStdioStream[pb.SandboxStdioReadV2Response]{}
	stub := &releasedStdioClient{sandboxStale: stale, sandboxReplacement: replacement}
	client := newStubClient(t, stub)
	reader := client.SandboxStdioReadV2(t.Context(), "ta-1", pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT).(*sandboxStdioReader)

	assertStdioReaderReopensReleasedStreamWithoutRetry(
		t, client, reader, &reader.stdioRetry, &stale.recvCalls, &replacement.recvCalls, &stub.offsets,
	)
}

func TestExecStdioReadReopensReleasedStreamWithoutRetry(t *testing.T) {
	t.Parallel()

	stale := &guardedStdioStream[pb.TaskExecStdioReadResponse]{
		response: pb.TaskExecStdioReadResponse_builder{Data: []byte("a")}.Build(),
	}
	replacement := &eofStdioStream[pb.TaskExecStdioReadResponse]{}
	stub := &releasedStdioClient{execStale: stale, execReplacement: replacement}
	client := newStubClient(t, stub)
	reader := client.ExecStdioRead(
		t.Context(), "ta-1", "ex-1", pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT, nil,
	).(*execStdioReader)

	assertStdioReaderReopensReleasedStreamWithoutRetry(
		t, client, reader, &reader.stdioRetry, &stale.recvCalls, &replacement.recvCalls, &stub.offsets,
	)
}

func TestSandboxStdioReadV2DropsEvictedPrefix(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// The worker evicted the first 1500 bytes, so only the retained tail is delivered.
	fake := &fakeStdioRouterServer{stdoutData: deterministicBytes(5000), chunkSize: 1000, droppedPrefix: 1500}
	client := newStdioTestClient(t, fake)

	reader := client.SandboxStdioReadV2(t.Context(), "ta-1", pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT)
	defer func() { _ = reader.Close() }()

	got, err := io.ReadAll(reader)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal(fake.stdoutData[1500:]))
}

func TestSandboxStdinWriteV2(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	fake := &fakeStdioRouterServer{}
	client := newStdioTestClient(t, fake)

	err := client.SandboxStdinWriteV2(t.Context(), "ta-1", 0, []byte("hello"), false)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	err = client.SandboxStdinWriteV2(t.Context(), "ta-1", 5, nil, true)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	received, closed := fake.stdinSnapshot()
	g.Expect(received).To(gomega.Equal([]byte("hello")))
	g.Expect(closed).To(gomega.BeTrue())
}

// A connection nobody ever uses still has to go back: the countdown starts when
// the client is built, not when its first operation finishes.
func TestUnusedCommandRouterClientReleasesItsConnection(t *testing.T) {
	g := gomega.NewWithT(t)

	client, err := newTaskCommandRouterClient(commandRouterParams{
		taskID:      "ta-1",
		jwt:         mockJWT(time.Now().Unix() + 3600),
		target:      "passthrough:///unused",
		creds:       insecure.NewCredentials(),
		idleTimeout: 50 * time.Millisecond,
		logger:      slog.New(slog.DiscardHandler),
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = client.Close() })

	released := func() bool {
		client.connMu.Lock()
		defer client.connMu.Unlock()
		return client.conn == nil
	}
	g.Eventually(released, time.Second, 10*time.Millisecond).Should(gomega.BeTrue(),
		"a client nothing ever used should have given its connection back")
}
