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

		if opts.MaxRetries != nil && numRetries >= *opts.MaxRetries {
			return nil, err
		}

		if opts.Deadline != nil && time.Now().Add(delay).After(*opts.Deadline) {
			return nil, errDeadlineExceeded
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
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
	logger *slog.Logger,
	profile Profile,
) (*taskCommandRouterClient, error) {
	resp, err := serverClient.TaskGetCommandRouterAccess(ctx, pb.TaskGetCommandRouterAccessRequest_builder{
		TaskId: taskID,
	}.Build())
	if err != nil {
		return nil, err
	}

	logger.DebugContext(ctx, "Using command router access for task", "task_id", taskID, "url", resp.GetUrl())

	jwt := resp.GetJwt()
	jwtExp, err := parseJwtExpiration(jwt)
	if err != nil {
		return nil, fmt.Errorf("parseJwtExpiration: %w", err)
	}

	url, err := url.Parse(resp.GetUrl())
	if err != nil {
		return nil, fmt.Errorf("failed to parse task router URL: %w", err)
	}

	if url.Scheme != "https" {
		return nil, fmt.Errorf("task router URL must be https, got: %s", resp.GetUrl())
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
		serverURL:    resp.GetUrl(),
		logger:       logger,
	}
	client.jwt.Store(&jwt)
	client.jwtExp.Store(jwtExp)

	logger.DebugContext(ctx, "Successfully initialized command router client", "task_id", taskID)
	return client, nil
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

		resp, err := c.serverClient.TaskGetCommandRouterAccess(ctx, pb.TaskGetCommandRouterAccessRequest_builder{
			TaskId: c.taskID,
		}.Build())
		if err != nil {
			return nil, fmt.Errorf("failed to refresh JWT: %w", err)
		}

		if resp.GetUrl() != c.serverURL {
			c.logger.WarnContext(ctx, "Task router URL changed during session")
		}

		jwt := resp.GetJwt()
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

// MountDirectory mounts an image at a directory in the container.
func (c *taskCommandRouterClient) MountDirectory(ctx context.Context, request *pb.TaskMountDirectoryRequest) error {
	_, err := callWithRetriesOnTransientErrors(ctx, func() (*emptypb.Empty, error) {
		return callWithAuthRetry(ctx, c, func(authCtx context.Context) (*emptypb.Empty, error) {
			return c.stub.TaskMountDirectory(authCtx, request)
		})
	}, defaultRetryOptions(), &c.closed)
	return err
}

// UnmountDirectory unmounts a directory in the container.
func (c *taskCommandRouterClient) UnmountDirectory(ctx context.Context, request *pb.TaskUnmountDirectoryRequest) error {
	_, err := callWithRetriesOnTransientErrors(ctx, func() (*emptypb.Empty, error) {
		return callWithAuthRetry(ctx, c, func(authCtx context.Context) (*emptypb.Empty, error) {
			return c.stub.TaskUnmountDirectory(authCtx, request)
		})
	}, defaultRetryOptions(), &c.closed)
	return err
}

// SnapshotDirectory snapshots a directory into a new image.
func (c *taskCommandRouterClient) SnapshotDirectory(ctx context.Context, request *pb.TaskSnapshotDirectoryRequest) (*pb.TaskSnapshotDirectoryResponse, error) {
	return callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskSnapshotDirectoryResponse, error) {
		return callWithAuthRetry(ctx, c, func(authCtx context.Context) (*pb.TaskSnapshotDirectoryResponse, error) {
			return c.stub.TaskSnapshotDirectory(authCtx, request)
		})
	}, defaultRetryOptions(), &c.closed)
}

// ExecStart starts a command execution.
func (c *taskCommandRouterClient) ExecStart(ctx context.Context, request *pb.TaskExecStartRequest) (*pb.TaskExecStartResponse, error) {
	return callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskExecStartResponse, error) {
		return callWithAuthRetry(ctx, c, func(authCtx context.Context) (*pb.TaskExecStartResponse, error) {
			return c.stub.TaskExecStart(authCtx, request)
		})
	}, defaultRetryOptions(), &c.closed)
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

	_, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskExecStdinWriteResponse, error) {
		return callWithAuthRetry(ctx, c, func(authCtx context.Context) (*pb.TaskExecStdinWriteResponse, error) {
			return c.stub.TaskExecStdinWrite(authCtx, request)
		})
	}, defaultRetryOptions(), &c.closed)
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
