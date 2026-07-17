package modal

import (
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/singleflight"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

// retryOptions configures retry behavior for callWithRetriesOnTransientErrors.
type retryOptions struct {
	BaseDelay   time.Duration
	DelayFactor float64
	MaxRetries  *int // nil means retry forever
	Deadline    *time.Time
	// ExcludeCodes lists gRPC status codes to exclude from retries even
	// if they would otherwise be retryable. Use this to let errors like
	// DeadlineExceeded propagate immediately when the caller has
	// specified their own deadline.
	ExcludeCodes []codes.Code
}

// defaultRetryOptions returns the default retry options.
func defaultRetryOptions() retryOptions {
	maxRetries := 10
	return retryOptions{
		BaseDelay:   10 * time.Millisecond,
		DelayFactor: 2.0,
		MaxRetries:  &maxRetries,
		Deadline:    nil,
	}
}

var commandRouterRetryableCodes = map[codes.Code]struct{}{
	codes.DeadlineExceeded: {},
	codes.Unavailable:      {},
	codes.Canceled:         {},
	codes.Internal:         {},
	codes.Unknown:          {},
}

// streamingStdinChunkSize is the number of bytes per outbound stdin stream
// message. It bounds the per-failure resend cost while amortizing per-chunk
// overhead.
const streamingStdinChunkSize = 256 * 1024

// streamingStdinMaxResumeAttempts caps resume retries for ExecStdinWriteStream:
// 10 total attempts, matching the unary path's retry budget.
const streamingStdinMaxResumeAttempts = 9

// parseJwtExpiration extracts the expiration time from a JWT token.
// Returns (nil, nil) if the token has no exp claim.
// Returns an error if the token is malformed.
func parseJwtExpiration(jwt string) (*int64, error) {
	parts := strings.Split(jwt, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("malformed JWT: expected 3 parts, got %d", len(parts))
	}

	payloadB64 := parts[1]
	switch len(payloadB64) % 4 {
	case 2:
		payloadB64 += "=="
	case 3:
		payloadB64 += "="
	}

	payloadJSON, err := base64.URLEncoding.DecodeString(payloadB64)
	if err != nil {
		return nil, fmt.Errorf("malformed JWT: base64 decode: %w", err)
	}

	var payload struct {
		Exp json.Number `json:"exp"`
	}
	if err := json.Unmarshal(payloadJSON, &payload); err != nil {
		return nil, fmt.Errorf("malformed JWT: json unmarshal: %w", err)
	}

	if payload.Exp == "" {
		return nil, nil
	}

	exp, err := payload.Exp.Int64()
	if err != nil {
		return nil, fmt.Errorf("malformed JWT: exp not an integer: %w", err)
	}

	return &exp, nil
}

var errDeadlineExceeded = errors.New("deadline exceeded")

// callWithRetriesOnTransientErrors retries the given function on transient gRPC errors.
func callWithRetriesOnTransientErrors[T any](
	ctx context.Context,
	fn func() (*T, error),
	opts retryOptions,
	closed *atomic.Bool,
) (*T, error) {
	delay := opts.BaseDelay
	numRetries := 0

	for {
		if opts.Deadline != nil && time.Now().After(*opts.Deadline) {
			return nil, errDeadlineExceeded
		}

		result, err := fn()
		if err == nil {
			return result, nil
		}

		st, ok := status.FromError(err)
		if !ok {
			return nil, err
		}
		if closed != nil && closed.Load() && st.Code() == codes.Canceled {
			return nil, ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
		}

		if _, retryable := commandRouterRetryableCodes[st.Code()]; !retryable {
			return nil, err
		}
		for _, excluded := range opts.ExcludeCodes {
			if excluded == st.Code() {
				return nil, err
			}
		}

		if opts.MaxRetries != nil && numRetries >= *opts.MaxRetries {
			return nil, err
		}

		// Clamp the backoff to the remaining deadline budget so we don't
		// sleep past the deadline. If the budget is already exhausted, the
		// next iteration's top-of-loop check returns errDeadlineExceeded
		// with `time.Now()` actually past the deadline — letting callers
		// translate consistently against the wall clock.
		sleepFor := delay
		if opts.Deadline != nil {
			if remaining := time.Until(*opts.Deadline); remaining < sleepFor {
				sleepFor = remaining
			}
		}
		if sleepFor < 0 {
			sleepFor = 0
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(sleepFor):
		}

		delay = time.Duration(float64(delay) * opts.DelayFactor)
		numRetries++
	}
}

// taskCommandRouterClient provides a client for the TaskCommandRouter gRPC service.
type taskCommandRouterClient struct {
	stub         pb.TaskCommandRouterClient
	conn         *grpc.ClientConn
	serverClient pb.ModalClientClient
	taskID       string
	sandboxID    string
	isV2         bool
	serverURL    string
	jwt          atomic.Pointer[string]
	jwtExp       atomic.Pointer[int64]
	logger       *slog.Logger
	closed       atomic.Bool

	refreshJwtGroup singleflight.Group
}

// initTaskCommandRouterClient attempts to initialize a TaskCommandRouterClient.
// Returns nil if command router access is not available for this task.
func initTaskCommandRouterClient(
	ctx context.Context,
	serverClient pb.ModalClientClient,
	taskID string,
	sandboxID string,
	isV2 bool,
	logger *slog.Logger,
	profile Profile,
) (*taskCommandRouterClient, error) {
	resp, err := getCommandRouterAccess(ctx, serverClient, taskID, sandboxID, isV2)
	if err != nil {
		return nil, err
	}

	logger.DebugContext(ctx, "Using command router access for task", "task_id", taskID, "url", resp.url)

	jwt := resp.jwt
	jwtExp, err := parseJwtExpiration(jwt)
	if err != nil {
		return nil, fmt.Errorf("parseJwtExpiration: %w", err)
	}

	url, err := url.Parse(resp.url)
	if err != nil {
		return nil, fmt.Errorf("failed to parse task router URL: %w", err)
	}

	if url.Scheme != "https" {
		return nil, fmt.Errorf("task router URL must be https, got: %s", resp.url)
	}

	host := url.Hostname()
	port := url.Port()
	if port == "" {
		port = "443"
	}
	target := fmt.Sprintf("%s:%s", host, port)

	var creds credentials.TransportCredentials
	if profile.isLocalhost() {
		logger.WarnContext(ctx, "Using insecure TLS (skip certificate verification) for task command router")
		creds = insecure.NewCredentials()
	} else {
		creds = credentials.NewTLS(&tls.Config{})

	}

	conn, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(creds),
		grpc.WithInitialWindowSize(windowSize),
		grpc.WithInitialConnWindowSize(windowSize),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMessageSize),
			grpc.MaxCallSendMsgSize(maxMessageSize),
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create task command router connection: %w", err)
	}

	client := &taskCommandRouterClient{
		stub:         pb.NewTaskCommandRouterClient(conn),
		conn:         conn,
		serverClient: serverClient,
		taskID:       taskID,
		sandboxID:    sandboxID,
		isV2:         isV2,
		serverURL:    resp.url,
		logger:       logger,
	}
	client.jwt.Store(&jwt)
	client.jwtExp.Store(jwtExp)

	logger.DebugContext(ctx, "Successfully initialized command router client", "task_id", taskID)
	return client, nil
}

type commandRouterAccess struct {
	jwt string
	url string
}

func getCommandRouterAccess(
	ctx context.Context,
	serverClient pb.ModalClientClient,
	taskID string,
	sandboxID string,
	isV2 bool,
) (*commandRouterAccess, error) {
	if isV2 {
		resp, err := serverClient.SandboxGetCommandRouterAccess(ctx, pb.SandboxGetCommandRouterAccessRequest_builder{
			SandboxId: &sandboxID,
		}.Build())
		if err != nil {
			return nil, err
		}
		return &commandRouterAccess{jwt: resp.GetJwt(), url: resp.GetUrl()}, nil
	}

	resp, err := serverClient.TaskGetCommandRouterAccess(ctx, pb.TaskGetCommandRouterAccessRequest_builder{
		TaskId: taskID,
	}.Build())
	if err != nil {
		return nil, err
	}
	return &commandRouterAccess{jwt: resp.GetJwt(), url: resp.GetUrl()}, nil
}

// Close closes the gRPC connection and cancels all in-flight operations.
func (c *taskCommandRouterClient) Close() error {
	if !c.closed.CompareAndSwap(false, true) {
		return nil
	}
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

func (c *taskCommandRouterClient) authContext(ctx context.Context) context.Context {
	return metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+*c.jwt.Load())
}

func (c *taskCommandRouterClient) refreshJwt(ctx context.Context) error {
	const jwtRefreshBufferSeconds = 30

	if c.closed.Load() {
		return errors.New("client is closed")
	}

	// If the current JWT expiration is already far enough in the future, don't refresh.
	if exp := c.jwtExp.Load(); exp != nil && *exp-time.Now().Unix() > jwtRefreshBufferSeconds {
		c.logger.DebugContext(ctx, "Skipping JWT refresh because expiration is far enough in the future", "task_id", c.taskID)
		return nil
	}

	_, err, _ := c.refreshJwtGroup.Do("refresh", func() (any, error) {
		if exp := c.jwtExp.Load(); exp != nil && *exp-time.Now().Unix() > jwtRefreshBufferSeconds {
			return nil, nil
		}

		resp, err := getCommandRouterAccess(ctx, c.serverClient, c.taskID, c.sandboxID, c.isV2)
		if err != nil {
			return nil, fmt.Errorf("failed to refresh JWT: %w", err)
		}

		if resp.url != c.serverURL {
			c.logger.WarnContext(ctx, "Task router URL changed during session")
		}

		jwt := resp.jwt
		c.jwt.Store(&jwt)
		jwtExp, err := parseJwtExpiration(jwt)
		if err != nil {
			// Log warning but continue - we'll refresh on every auth failure instead of proactively.
			c.logger.WarnContext(ctx, "parseJwtExpiration during refresh", "error", err)
		}
		c.jwtExp.Store(jwtExp)
		return nil, nil
	})
	return err
}

type retryableClient interface {
	authContext(ctx context.Context) context.Context
	refreshJwt(ctx context.Context) error
}

func callWithAuthRetry[T any](ctx context.Context, c retryableClient, fn func(context.Context) (*T, error)) (*T, error) {
	resp, err := fn(c.authContext(ctx))
	if err != nil {
		if st, ok := status.FromError(err); ok && st.Code() == codes.Unauthenticated {
			if refreshErr := c.refreshJwt(ctx); refreshErr != nil {
				return nil, refreshErr
			}
			return fn(c.authContext(ctx))
		}
	}
	return resp, err
}

func callCommandRouterUnary[T any](ctx context.Context, c *taskCommandRouterClient, fn func(context.Context) (*T, error)) (*T, error) {
	return callWithRetriesOnTransientErrors(ctx, func() (*T, error) {
		return callWithAuthRetry(ctx, c, fn)
	}, defaultRetryOptions(), &c.closed)
}

// SetNetworkAccess replaces the task's outbound network allowlist (domains + CIDRs).
func (c *taskCommandRouterClient) SetNetworkAccess(ctx context.Context, request *pb.TaskSetNetworkAccessRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskSetNetworkAccessResponse, error) {
		return c.stub.TaskSetNetworkAccess(authCtx, request)
	})
	return err
}

// MountDirectory mounts an image at a directory in the container.
func (c *taskCommandRouterClient) MountDirectory(ctx context.Context, request *pb.TaskMountDirectoryRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*emptypb.Empty, error) {
		return c.stub.TaskMountDirectory(authCtx, request)
	})
	return err
}

// UnmountDirectory unmounts a directory in the container.
func (c *taskCommandRouterClient) UnmountDirectory(ctx context.Context, request *pb.TaskUnmountDirectoryRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*emptypb.Empty, error) {
		return c.stub.TaskUnmountDirectory(authCtx, request)
	})
	return err
}

// ReloadVolumes reloads all Volumes mounted in the task to reflect their latest committed state.
//
// timeout is the client-side deadline. If the reload does not complete within
// this window, the call is cancelled and a TimeoutError is returned.
func (c *taskCommandRouterClient) ReloadVolumes(ctx context.Context, request *pb.TaskReloadVolumesRequest, timeout time.Duration) error {
	overallDeadline := time.Now().Add(timeout)
	opts := defaultRetryOptions()
	opts.ExcludeCodes = []codes.Code{codes.DeadlineExceeded, codes.Canceled}
	opts.Deadline = &overallDeadline
	_, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskReloadVolumesResponse, error) {
		remaining := time.Until(overallDeadline)
		callCtx, cancel := context.WithTimeout(ctx, remaining)
		defer cancel()
		return callWithAuthRetry(callCtx, c, func(authCtx context.Context) (*pb.TaskReloadVolumesResponse, error) {
			return c.stub.TaskReloadVolumes(authCtx, request)
		})
	}, opts, &c.closed)
	if err != nil && time.Now().After(overallDeadline) {
		return TimeoutError{Exception: "Timeout expired"}
	}
	return err
}

// SnapshotDirectory snapshots a directory into a new image.
//
// Mirrors SnapshotFilesystem: `timeout` is the overall budget across all
// retry attempts. Each attempt receives the *remaining* budget as its
// per-call gRPC deadline; any error observed at or after the deadline
// is translated into a TimeoutError. Errors observed *before* the
// deadline (including a caller-driven ctx cancellation) are propagated
// unchanged.
func (c *taskCommandRouterClient) SnapshotDirectory(ctx context.Context, request *pb.TaskSnapshotDirectoryRequest, timeout time.Duration) (*pb.TaskSnapshotDirectoryResponse, error) {
	overallDeadline := time.Now().Add(timeout)
	opts := defaultRetryOptions()
	opts.ExcludeCodes = []codes.Code{codes.DeadlineExceeded, codes.Canceled}
	opts.Deadline = &overallDeadline
	resp, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskSnapshotDirectoryResponse, error) {
		remaining := time.Until(overallDeadline)
		callCtx, cancel := context.WithTimeout(ctx, remaining)
		defer cancel()
		return callWithAuthRetry(callCtx, c, func(authCtx context.Context) (*pb.TaskSnapshotDirectoryResponse, error) {
			return c.stub.TaskSnapshotDirectory(authCtx, request)
		})
	}, opts, &c.closed)
	if err != nil && time.Now().After(overallDeadline) {
		return nil, TimeoutError{Exception: "Timeout expired"}
	}
	return resp, err
}

// SnapshotFilesystem snapshots the full container filesystem into a new image.
//
// `timeout` is the overall budget across all retry attempts: each
// attempt receives the *remaining* budget as its per-call gRPC
// deadline, and retries are aborted once the deadline elapses (rather
// than granting another fresh full window). DeadlineExceeded / Canceled
// responses are excluded from retries so the deadline isn't reset by
// another attempt.
//
// Any error observed at or after the deadline is translated into a
// TimeoutError. Errors observed *before* the deadline are propagated
// unchanged — that includes the caller's own ctx cancellation (which
// grpc-go surfaces as codes.Canceled), so callers see their cancel
// rather than a misleading timeout.
func (c *taskCommandRouterClient) SnapshotFilesystem(ctx context.Context, request *pb.TaskSnapshotFilesystemRequest, timeout time.Duration) (*pb.TaskSnapshotFilesystemResponse, error) {
	overallDeadline := time.Now().Add(timeout)
	opts := defaultRetryOptions()
	opts.ExcludeCodes = []codes.Code{codes.DeadlineExceeded, codes.Canceled}
	opts.Deadline = &overallDeadline
	resp, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskSnapshotFilesystemResponse, error) {
		// Per-call timeout = remaining budget on the overall deadline.
		// A zero or negative remaining time would still create a usable
		// (already-expired) context, which grpc-go reports as DeadlineExceeded.
		remaining := time.Until(overallDeadline)
		callCtx, cancel := context.WithTimeout(ctx, remaining)
		defer cancel()
		return callWithAuthRetry(callCtx, c, func(authCtx context.Context) (*pb.TaskSnapshotFilesystemResponse, error) {
			return c.stub.TaskSnapshotFilesystem(authCtx, request)
		})
	}, opts, &c.closed)
	if err != nil && time.Now().After(overallDeadline) {
		return nil, TimeoutError{Exception: "Timeout expired"}
	}
	return resp, err
}

// SandboxWaitUntilReady waits until the Sandbox's readiness probe reports ready.
func (c *taskCommandRouterClient) SandboxWaitUntilReady(ctx context.Context, taskID string, timeout time.Duration) (*pb.SandboxWaitUntilReadyTcrResponse, error) {
	opts := defaultRetryOptions()
	overallDeadline := time.Now().Add(timeout)
	opts.Deadline = &overallDeadline

	resp, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.SandboxWaitUntilReadyTcrResponse, error) {
		remaining := max(time.Until(overallDeadline), time.Millisecond)
		request := pb.SandboxWaitUntilReadyTcrRequest_builder{
			TaskId:  taskID,
			Timeout: float32(remaining.Seconds()),
		}.Build()
		callCtx, cancel := context.WithTimeout(ctx, remaining)
		defer cancel()
		return callWithAuthRetry(callCtx, c, func(authCtx context.Context) (*pb.SandboxWaitUntilReadyTcrResponse, error) {
			return c.stub.SandboxWaitUntilReady(authCtx, request)
		})
	}, opts, &c.closed)
	if err != nil {
		if errors.Is(err, errDeadlineExceeded) {
			return nil, TimeoutError{Exception: "Timeout expired"}
		}
		return nil, err
	}
	return resp, nil
}

// ContainerCreate creates an additional container in the task.
func (c *taskCommandRouterClient) ContainerCreate(ctx context.Context, request *pb.TaskContainerCreateRequest) (*pb.TaskContainerCreateResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerCreateResponse, error) {
		return c.stub.TaskContainerCreate(authCtx, request)
	})
}

// ContainerGet returns the latest container associated with a logical name.
func (c *taskCommandRouterClient) ContainerGet(ctx context.Context, request *pb.TaskContainerGetRequest) (*pb.TaskContainerGetResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerGetResponse, error) {
		return c.stub.TaskContainerGet(authCtx, request)
	})
}

// ContainerList lists containers associated with the task.
func (c *taskCommandRouterClient) ContainerList(ctx context.Context, request *pb.TaskContainerListRequest) (*pb.TaskContainerListResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerListResponse, error) {
		return c.stub.TaskContainerList(authCtx, request)
	})
}

// ContainerTerminate terminates a tracked container.
func (c *taskCommandRouterClient) ContainerTerminate(ctx context.Context, request *pb.TaskContainerTerminateRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerTerminateResponse, error) {
		return c.stub.TaskContainerTerminate(authCtx, request)
	})
	return err
}

// ContainerWait waits for a tracked container to reach a terminal result.
func (c *taskCommandRouterClient) ContainerWait(ctx context.Context, request *pb.TaskContainerWaitRequest) (*pb.TaskContainerWaitResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerWaitResponse, error) {
		return c.stub.TaskContainerWait(authCtx, request)
	})
}

// ExecStart starts a command execution.
func (c *taskCommandRouterClient) ExecStart(ctx context.Context, request *pb.TaskExecStartRequest) (*pb.TaskExecStartResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskExecStartResponse, error) {
		return c.stub.TaskExecStart(authCtx, request)
	})
}

// ExecStdinWrite writes data to stdin of an exec.
func (c *taskCommandRouterClient) ExecStdinWrite(ctx context.Context, taskID, execID string, offset uint64, data []byte, eof bool) error {
	request := pb.TaskExecStdinWriteRequest_builder{
		TaskId: taskID,
		ExecId: execID,
		Offset: offset,
		Data:   data,
		Eof:    eof,
	}.Build()

	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskExecStdinWriteResponse, error) {
		return c.stub.TaskExecStdinWrite(authCtx, request)
	})
	return err
}

// ExecStdinStatus returns the current stdin write status for an exec'd
// command, to support resuming a stdin stream from the right offset.
//
// Evicts any in-flight stdin stream for the exec.
func (c *taskCommandRouterClient) ExecStdinStatus(ctx context.Context, taskID, execID string) (*pb.TaskExecStdinStatusResponse, error) {
	request := pb.TaskExecStdinStatusRequest_builder{
		TaskId: taskID,
		ExecId: execID,
	}.Build()

	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskExecStdinStatusResponse, error) {
		return c.stub.TaskExecStdinStatus(authCtx, request)
	})
}

// isStreamingStdinResumableCode reports whether a stdin stream attempt that
// failed with the given status code can be resumed. Adds Unauthenticated
// to the normal retryable codes, which is handled by refreshing the JWT
// between attempts rather than retrying in-stream.
func isStreamingStdinResumableCode(code codes.Code) bool {
	if code == codes.Unauthenticated {
		return true
	}
	_, ok := commandRouterRetryableCodes[code]
	return ok
}

// ExecStdinWriteStream streams source into the exec's stdin, with bounded
// resume on transient failures.
//
// On a resumable error, it queries ExecStdinStatus for
// the server's offset, seeks source to that point, and reopens the
// stream.
// Returns the total number of bytes streamed.
func (c *taskCommandRouterClient) ExecStdinWriteStream(ctx context.Context, taskID, execID string, source io.ReadSeeker) (int64, error) {
	var offset uint64
	attempt := 0
	for {
		if _, err := source.Seek(int64(offset), io.SeekStart); err != nil {
			return 0, err
		}
		attemptErr := c.execStdinWriteStreamAttempt(ctx, taskID, execID, offset, source)
		if attemptErr == nil {
			return source.Seek(0, io.SeekCurrent)
		}

		st, ok := status.FromError(attemptErr)
		if !ok {
			// Non-status errors (e.g. local source read failures) are not resumable.
			return 0, attemptErr
		}
		if c.closed.Load() && st.Code() == codes.Canceled {
			return 0, ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
		}
		if !isStreamingStdinResumableCode(st.Code()) {
			return 0, attemptErr
		}
		attempt++
		if attempt > streamingStdinMaxResumeAttempts {
			return 0, attemptErr
		}
		// There is no in-stream auth retry: refresh the JWT here so the next
		// resume attempt opens its stream with a fresh token.
		if st.Code() == codes.Unauthenticated {
			if refreshErr := c.refreshJwt(ctx); refreshErr != nil {
				return 0, refreshErr
			}
		}
		statusResp, statusErr := c.ExecStdinStatus(ctx, taskID, execID)
		if statusErr != nil {
			return 0, statusErr
		}
		if statusResp.GetClosed() {
			// If the server's byte count matches everything we streamed and the source is
			// exhausted, the upload completed and only the response was lost.
			currentPos, err := source.Seek(0, io.SeekCurrent)
			if err != nil {
				return 0, err
			}
			if statusResp.GetNumBytesWritten() == uint64(currentPos) {
				endPos, seekErr := source.Seek(0, io.SeekEnd)
				if seekErr != nil {
					return 0, seekErr
				}
				if endPos == currentPos {
					c.logger.DebugContext(ctx, "ExecStdinWriteStream completed but response was lost", "exec_id", execID, "error", attemptErr)
					return currentPos, nil
				}
			}
			return 0, attemptErr
		}
		offset = statusResp.GetNumBytesWritten()
		c.logger.DebugContext(ctx, "ExecStdinWriteStream resuming", "exec_id", execID, "offset", offset, "error", attemptErr)
	}
}

// execStdinWriteStreamAttempt performs a single client-streaming attempt:
// Start, Data chunks, then End (EOF). It does not retry; ExecStdinWriteStream
// owns resume.
func (c *taskCommandRouterClient) execStdinWriteStreamAttempt(ctx context.Context, taskID, execID string, offset uint64, source io.Reader) error {
	// Cancel the stream when bailing out before CloseAndRecv completes so an
	// abandoned attempt doesn't leak. A canceled stream ends without End,
	// which leaves stdin open server-side for resume.
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := c.stub.TaskExecStdinWriteStream(c.authContext(attemptCtx))
	if err != nil {
		return err
	}

	// When Send fails because the stream was aborted remotely, it returns
	// io.EOF and the actual result must be retrieved from CloseAndRecv. A nil
	// CloseAndRecv error means the server completed the RPC successfully, so
	// the attempt succeeded even though the final sends were not consumed.
	sendErrStatus := func(sendErr error) error {
		if errors.Is(sendErr, io.EOF) {
			_, recvErr := stream.CloseAndRecv()
			return recvErr
		}
		return sendErr
	}

	start := pb.TaskExecStdinWriteStreamRequest_builder{
		Start: pb.TaskExecStdinWriteStreamStart_builder{
			TaskId: taskID,
			ExecId: execID,
			Offset: offset,
		}.Build(),
	}.Build()
	if err := stream.Send(start); err != nil {
		return sendErrStatus(err)
	}

	buf := make([]byte, streamingStdinChunkSize)
	for {
		n, readErr := source.Read(buf)
		if n > 0 {
			msg := pb.TaskExecStdinWriteStreamRequest_builder{Data: buf[:n]}.Build()
			if err := stream.Send(msg); err != nil {
				return sendErrStatus(err)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}

	// The server closes the exec's stdin only on this explicit End message.
	// A stream that breaks before it leaves stdin open for resume.
	end := pb.TaskExecStdinWriteStreamRequest_builder{
		End: &pb.TaskExecStdinWriteStreamEnd{},
	}.Build()
	if err := stream.Send(end); err != nil {
		return sendErrStatus(err)
	}
	_, err = stream.CloseAndRecv()
	return err
}

// ExecWait waits for an exec to complete and returns the exit code.
func (c *taskCommandRouterClient) ExecWait(ctx context.Context, taskID, execID string, deadline *time.Time) (*pb.TaskExecWaitResponse, error) {
	request := pb.TaskExecWaitRequest_builder{
		TaskId: taskID,
		ExecId: execID,
	}.Build()

	if deadline != nil && time.Now().After(*deadline) {
		return nil, ExecTimeoutError{Exception: fmt.Sprintf("deadline exceeded while waiting for exec %s", execID)}
	}

	opts := retryOptions{
		BaseDelay:   1 * time.Second, // Retry after 1s since total time is expected to be long.
		DelayFactor: 1,               // Fixed delay.
		MaxRetries:  nil,             // Retry forever.
		Deadline:    deadline,
	}

	resp, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskExecWaitResponse, error) {
		return callWithAuthRetry(ctx, c, func(authCtx context.Context) (*pb.TaskExecWaitResponse, error) {
			// Set a per-call timeout of 60 seconds
			callCtx, cancel := context.WithTimeout(authCtx, 60*time.Second)
			defer cancel()
			return c.stub.TaskExecWait(callCtx, request)
		})
	}, opts, &c.closed)

	if err != nil {
		st, ok := status.FromError(err)
		if (ok && st.Code() == codes.DeadlineExceeded) || errors.Is(err, errDeadlineExceeded) {
			return nil, ExecTimeoutError{Exception: fmt.Sprintf("deadline exceeded while waiting for exec %s", execID)}
		}
	}
	return resp, err
}

// stdioReadResult represents a result from the stdio read stream.
type stdioReadResult struct {
	Response *pb.TaskExecStdioReadResponse
	Err      error
}

// ExecStdioRead reads stdout or stderr from an exec.
// The returned channel will be closed when the stream ends or an error occurs.
func (c *taskCommandRouterClient) ExecStdioRead(
	ctx context.Context,
	taskID, execID string,
	fd pb.FileDescriptor,
	deadline *time.Time,
) <-chan stdioReadResult {
	resultCh := make(chan stdioReadResult)

	go func() {
		defer close(resultCh)

		var srFd pb.TaskExecStdioFileDescriptor
		switch fd {
		case pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT:
			srFd = pb.TaskExecStdioFileDescriptor_TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT
		case pb.FileDescriptor_FILE_DESCRIPTOR_STDERR:
			srFd = pb.TaskExecStdioFileDescriptor_TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDERR
		case pb.FileDescriptor_FILE_DESCRIPTOR_INFO, pb.FileDescriptor_FILE_DESCRIPTOR_UNSPECIFIED:
			resultCh <- stdioReadResult{Err: fmt.Errorf("unsupported file descriptor: %v", fd)}
			return
		default:
			resultCh <- stdioReadResult{Err: fmt.Errorf("invalid file descriptor: %v", fd)}
			return
		}

		if deadline != nil {
			var deadlineCancel context.CancelFunc
			ctx, deadlineCancel = context.WithDeadline(ctx, *deadline)
			defer deadlineCancel()
		}
		c.streamStdio(ctx, resultCh, taskID, execID, srFd)
	}()

	return resultCh
}

func (c *taskCommandRouterClient) streamStdio(
	ctx context.Context,
	resultCh chan<- stdioReadResult,
	taskID, execID string,
	fd pb.TaskExecStdioFileDescriptor,
) {
	deadline, hasDeadline := ctx.Deadline()

	var offset int64
	delayOriginal := 10 * time.Millisecond
	numRetriesRemainingOriginal := 10

	delayFactor := 2.0
	delay := delayOriginal
	numRetriesRemaining := numRetriesRemainingOriginal
	didAuthRetry := false

	for {
		if ctx.Err() != nil {
			if hasDeadline && ctx.Err() == context.DeadlineExceeded {
				resultCh <- stdioReadResult{Err: ExecTimeoutError{Exception: fmt.Sprintf("deadline exceeded while streaming stdio for exec %s", execID)}}
			} else {
				resultCh <- stdioReadResult{Err: ctx.Err()}
			}
			return
		}

		callCtx := c.authContext(ctx)

		request := pb.TaskExecStdioReadRequest_builder{
			TaskId:         taskID,
			ExecId:         execID,
			Offset:         uint64(offset),
			FileDescriptor: fd,
		}.Build()

		stream, err := c.stub.TaskExecStdioRead(callCtx, request)
		if err != nil {
			errStatus := status.Code(err)
			if errStatus == codes.Unauthenticated && !didAuthRetry {
				if refreshErr := c.refreshJwt(ctx); refreshErr != nil {
					resultCh <- stdioReadResult{Err: refreshErr}
					return
				}
				didAuthRetry = true
				continue
			}
			if c.closed.Load() && errStatus == codes.Canceled {
				closedErr := ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
				resultCh <- stdioReadResult{Err: closedErr}
				return
			}
			if _, retryable := commandRouterRetryableCodes[status.Code(err)]; retryable && numRetriesRemaining > 0 {
				if hasDeadline && time.Until(deadline) <= delay {
					resultCh <- stdioReadResult{Err: ExecTimeoutError{Exception: fmt.Sprintf("deadline exceeded while streaming stdio for exec %s", execID)}}
					return
				}
				c.logger.DebugContext(ctx, "Retrying stdio read with delay", "delay", delay, "error", err)
				select {
				case <-ctx.Done():
					resultCh <- stdioReadResult{Err: ctx.Err()}
					return
				case <-time.After(delay):
				}
				delay = time.Duration(float64(delay) * delayFactor)
				numRetriesRemaining--
				continue
			}
			resultCh <- stdioReadResult{Err: err}
			return
		}

		for {
			item, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				errStatus := status.Code(err)
				if errStatus == codes.Unauthenticated && !didAuthRetry {
					if refreshErr := c.refreshJwt(ctx); refreshErr != nil {
						resultCh <- stdioReadResult{Err: refreshErr}
						return
					}
					didAuthRetry = true
					break
				}
				if c.closed.Load() && errStatus == codes.Canceled {
					closedErr := ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
					resultCh <- stdioReadResult{Err: closedErr}
					return
				}
				if _, retryable := commandRouterRetryableCodes[errStatus]; retryable && numRetriesRemaining > 0 {
					if hasDeadline && time.Until(deadline) <= delay {
						resultCh <- stdioReadResult{Err: ExecTimeoutError{Exception: fmt.Sprintf("deadline exceeded while streaming stdio for exec %s", execID)}}
						return
					}
					c.logger.DebugContext(ctx, "Retrying stdio read with delay", "delay", delay, "error", err)
					select {
					case <-ctx.Done():
						resultCh <- stdioReadResult{Err: ctx.Err()}
						return
					case <-time.After(delay):
					}
					delay = time.Duration(float64(delay) * delayFactor)
					numRetriesRemaining--
					break
				}
				resultCh <- stdioReadResult{Err: err}
				return
			}

			if didAuthRetry {
				didAuthRetry = false
			}
			delay = delayOriginal
			numRetriesRemaining = numRetriesRemainingOriginal
			offset += int64(len(item.GetData()))

			resultCh <- stdioReadResult{Response: item}
		}
	}
}
