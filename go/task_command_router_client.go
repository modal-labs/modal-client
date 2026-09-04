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
	"sync"
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
	// connMu guards the fields below, not the connection itself: it is held
	// while the bookkeeping is read or written and released before anything
	// goes on the wire, so operations still run concurrently and inFlight
	// counts how many. Dialling happens under it, which is only cheap because
	// grpc.NewClient connects lazily.
	//
	// The connection is dropped when nothing has used it for idleTimeout and
	// dialled again by the next operation, so nothing may hold stubValue across
	// one.
	connMu    sync.RWMutex
	stubValue pb.TaskCommandRouterClient
	conn      *grpc.ClientConn
	inFlight  int
	idleTimer *time.Timer
	// Which timer the pending callback belongs to. A callback that finds a
	// different one has been superseded: it was already running when a new
	// operation replaced its timer, so it speaks for a connection that has
	// since been used.
	idleTimerSeq uint64
	target       string
	creds        credentials.TransportCredentials
	// How long the client may go unused before its connection is given up. Zero
	// keeps it up until the client is closed.
	idleTimeout time.Duration
	// Bumped every time a connection is dialled. A stream opened on an earlier
	// one is stale: the connection under it has since been given up.
	generation atomic.Uint64

	serverClient    pb.ModalClientClient
	taskID          string
	sandboxID       string
	isV2            bool
	serverURL       string
	jwt             atomic.Pointer[string]
	jwtExp          atomic.Pointer[int64]
	logger          *slog.Logger
	closed          atomic.Bool
	refreshJwtGroup singleflight.Group
}

// access is the router access returned by SandboxCreateV2, which lets a freshly
// created sandbox connect without a round-trip. Pass nil to look it up instead.
func initTaskCommandRouterClient(
	ctx context.Context,
	serverClient pb.ModalClientClient,
	taskID string,
	sandboxID string,
	isV2 bool,
	access *commandRouterAccess,
	logger *slog.Logger,
	profile Profile,
) (*taskCommandRouterClient, error) {
	if access == nil {
		var err error
		access, err = getCommandRouterAccess(ctx, serverClient, taskID, sandboxID, isV2)
		if err != nil {
			return nil, err
		}
	}

	logger.DebugContext(ctx, "Using command router access for task", "task_id", taskID, "url", access.url)

	jwt := access.jwt
	jwtExp, err := parseJwtExpiration(jwt)
	if err != nil {
		return nil, fmt.Errorf("parseJwtExpiration: %w", err)
	}

	url, err := url.Parse(access.url)
	if err != nil {
		return nil, fmt.Errorf("failed to parse task router URL: %w", err)
	}

	if url.Scheme != "https" {
		return nil, fmt.Errorf("task router URL must be https, got: %s", access.url)
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

	client, err := newTaskCommandRouterClient(commandRouterParams{
		serverClient: serverClient,
		taskID:       taskID,
		sandboxID:    sandboxID,
		isV2:         isV2,
		serverURL:    access.url,
		jwt:          jwt,
		jwtExp:       jwtExp,
		target:       target,
		creds:        creds,
		idleTimeout:  profile.SandboxChannelIdleTimeout,
		logger:       logger,
	})
	if err != nil {
		return nil, err
	}

	logger.DebugContext(ctx, "Successfully initialized command router client", "task_id", taskID)
	return client, nil
}

// commandRouterParams is everything a client needs to reach the command router,
// and to reach it again after giving its connection up.
type commandRouterParams struct {
	serverClient pb.ModalClientClient
	taskID       string
	sandboxID    string
	isV2         bool
	serverURL    string
	jwt          string
	jwtExp       *int64
	// Held so the connection can be rebuilt, not only opened.
	target      string
	creds       credentials.TransportCredentials
	idleTimeout time.Duration
	logger      *slog.Logger
}

// newTaskCommandRouterClient dials the command router and returns a client for
// it. A client gives its connection up once nothing has used it and dials again
// on the next operation, so it needs the means to do that from the start.
func newTaskCommandRouterClient(p commandRouterParams) (*taskCommandRouterClient, error) {
	if p.target == "" {
		return nil, fmt.Errorf("command router client for task %s needs a dial target", p.taskID)
	}

	conn, err := dialCommandRouter(p.target, p.creds)
	if err != nil {
		return nil, err
	}

	client := &taskCommandRouterClient{
		stubValue:    pb.NewTaskCommandRouterClient(conn),
		conn:         conn,
		target:       p.target,
		creds:        p.creds,
		idleTimeout:  p.idleTimeout,
		serverClient: p.serverClient,
		taskID:       p.taskID,
		sandboxID:    p.sandboxID,
		isV2:         p.isV2,
		serverURL:    p.serverURL,
		logger:       p.logger,
	}
	client.jwt.Store(&p.jwt)
	client.jwtExp.Store(p.jwtExp)
	// The connection is live from here on, so start the countdown now rather
	// than when the first operation finishes: an operation may never come, and
	// nothing else would give the connection back.
	client.connMu.Lock()
	client.armIdleTimerLocked()
	client.connMu.Unlock()
	return client, nil
}

// dialCommandRouter opens a connection to the command router. It is called
// again when an idle connection has been given up and a new one is needed.
func dialCommandRouter(target string, creds credentials.TransportCredentials) (*grpc.ClientConn, error) {
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
	return conn, nil
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
	c.connMu.Lock()
	defer c.connMu.Unlock()
	c.stopIdleTimerLocked()
	if c.conn != nil {
		conn := c.conn
		// The stub is left in place. Close does not wait for operations already
		// under way - that is the point of it - so one of them may still read
		// the stub, and a call on a closed connection is an error where a call
		// on a nil stub is a panic.
		c.conn = nil
		return conn.Close()
	}
	return nil
}

// stub is the RPC stub to use right now. It changes when an idle connection is
// rebuilt, so it must be read per use rather than held across one.
//
// The lock only makes the field readable intact: an interface value is two
// words, so a read racing a rebuild could see one of each. Staying off a
// released connection is the lease's job, not this lock's.
func (c *taskCommandRouterClient) stub() pb.TaskCommandRouterClient {
	c.connMu.RLock()
	defer c.connMu.RUnlock()
	return c.stubValue
}

// beginOp says the client is about to be used: it holds off the idle timer and
// dials again if the connection was already given up. Every call must be paired
// with endOp.
func (c *taskCommandRouterClient) beginOp() error {
	c.connMu.Lock()
	defer c.connMu.Unlock()

	if c.closed.Load() {
		return ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
	}
	c.stopIdleTimerLocked()
	if c.conn == nil {
		conn, err := dialCommandRouter(c.target, c.creds)
		if err != nil {
			return err
		}
		c.conn, c.stubValue = conn, pb.NewTaskCommandRouterClient(conn)
		c.generation.Add(1)
		c.logger.DebugContext(context.Background(), "Reconnected to the command router after an idle release", "task_id", c.taskID)
	}
	c.inFlight++
	return nil
}

// endOp says the caller is done. The last one out starts the clock on giving
// the connection back.
func (c *taskCommandRouterClient) endOp() {
	c.connMu.Lock()
	defer c.connMu.Unlock()

	c.inFlight--
	if c.inFlight > 0 {
		return
	}
	c.armIdleTimerLocked()
}

// armIdleTimerLocked starts the countdown to giving the connection back. The
// caller must hold connMu and must have nothing in flight.
func (c *taskCommandRouterClient) armIdleTimerLocked() {
	if c.idleTimeout <= 0 || c.conn == nil {
		return
	}
	c.stopIdleTimerLocked()
	c.idleTimerSeq++
	seq := c.idleTimerSeq
	c.idleTimer = time.AfterFunc(c.idleTimeout, func() { c.closeIfStillIdle(seq) })
}

// closeIfStillIdle gives the connection back, so a Sandbox nobody is using
// costs nothing at all - no socket, and none of the goroutines behind it. The
// next operation dials again.
func (c *taskCommandRouterClient) closeIfStillIdle(seq uint64) {
	c.connMu.Lock()
	defer c.connMu.Unlock()

	if seq != c.idleTimerSeq {
		return
	}
	c.idleTimer = nil
	if c.inFlight > 0 || c.conn == nil {
		return
	}
	c.logger.DebugContext(context.Background(), "Releasing the command router connection to an idle Sandbox", "task_id", c.taskID)
	conn := c.conn
	// Left in place for the same reason as in Close.
	c.conn = nil
	if err := conn.Close(); err != nil {
		c.logger.DebugContext(context.Background(), "Failed to close an idle command router connection", "error", err)
	}
}

func (c *taskCommandRouterClient) stopIdleTimerLocked() {
	// Bumped whether or not a timer is set, so a callback already past its own
	// Stop finds a sequence it does not match and stands down.
	c.idleTimerSeq++
	if c.idleTimer != nil {
		c.idleTimer.Stop()
		c.idleTimer = nil
	}
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
	// beginOp and endOp bracket a use of the connection. Every unary call
	// reaches this interface, so taking the lease here is what stops a method
	// reaching a connection that has been given up.
	beginOp() error
	endOp()
}

func callWithAuthRetry[T any](ctx context.Context, c retryableClient, fn func(context.Context) (*T, error)) (*T, error) {
	if err := c.beginOp(); err != nil {
		return nil, err
	}
	defer c.endOp()

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
		return c.stub().TaskSetNetworkAccess(authCtx, request)
	})
	return err
}

// MountDirectory mounts an image at a directory in the container.
func (c *taskCommandRouterClient) MountDirectory(ctx context.Context, request *pb.TaskMountDirectoryRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*emptypb.Empty, error) {
		return c.stub().TaskMountDirectory(authCtx, request)
	})
	return err
}

// UnmountDirectory unmounts a directory in the container.
func (c *taskCommandRouterClient) UnmountDirectory(ctx context.Context, request *pb.TaskUnmountDirectoryRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*emptypb.Empty, error) {
		return c.stub().TaskUnmountDirectory(authCtx, request)
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
			return c.stub().TaskReloadVolumes(authCtx, request)
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
			return c.stub().TaskSnapshotDirectory(authCtx, request)
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
			return c.stub().TaskSnapshotFilesystem(authCtx, request)
		})
	}, opts, &c.closed)
	if err != nil && time.Now().After(overallDeadline) {
		return nil, TimeoutError{Exception: "Timeout expired"}
	}
	return resp, err
}

// SnapshotMemory snapshots the memory and filesystem of the container.
func (c *taskCommandRouterClient) SnapshotMemory(ctx context.Context, request *pb.TaskSnapshotMemoryRequest, timeout time.Duration) (*pb.TaskSnapshotMemoryResponse, error) {
	overallDeadline := time.Now().Add(timeout)
	opts := defaultRetryOptions()
	opts.ExcludeCodes = []codes.Code{codes.DeadlineExceeded, codes.Canceled}
	opts.Deadline = &overallDeadline
	resp, err := callWithRetriesOnTransientErrors(ctx, func() (*pb.TaskSnapshotMemoryResponse, error) {
		remaining := time.Until(overallDeadline)
		callCtx, cancel := context.WithTimeout(ctx, remaining)
		defer cancel()
		return callWithAuthRetry(callCtx, c, func(authCtx context.Context) (*pb.TaskSnapshotMemoryResponse, error) {
			return c.stub().TaskSnapshotMemory(authCtx, request)
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
			return c.stub().SandboxWaitUntilReady(authCtx, request)
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
		return c.stub().TaskContainerCreate(authCtx, request)
	})
}

// ContainerGet returns the latest container associated with a logical name.
func (c *taskCommandRouterClient) ContainerGet(ctx context.Context, request *pb.TaskContainerGetRequest) (*pb.TaskContainerGetResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerGetResponse, error) {
		return c.stub().TaskContainerGet(authCtx, request)
	})
}

// ContainerList lists containers associated with the task.
func (c *taskCommandRouterClient) ContainerList(ctx context.Context, request *pb.TaskContainerListRequest) (*pb.TaskContainerListResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerListResponse, error) {
		return c.stub().TaskContainerList(authCtx, request)
	})
}

// ContainerTerminate terminates a tracked container.
func (c *taskCommandRouterClient) ContainerTerminate(ctx context.Context, request *pb.TaskContainerTerminateRequest) error {
	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerTerminateResponse, error) {
		return c.stub().TaskContainerTerminate(authCtx, request)
	})
	return err
}

// ContainerWait waits for a tracked container to reach a terminal result.
func (c *taskCommandRouterClient) ContainerWait(ctx context.Context, request *pb.TaskContainerWaitRequest) (*pb.TaskContainerWaitResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskContainerWaitResponse, error) {
		return c.stub().TaskContainerWait(authCtx, request)
	})
}

// ExecStart starts a command execution.
func (c *taskCommandRouterClient) ExecStart(ctx context.Context, request *pb.TaskExecStartRequest) (*pb.TaskExecStartResponse, error) {
	return callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.TaskExecStartResponse, error) {
		return c.stub().TaskExecStart(authCtx, request)
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
		return c.stub().TaskExecStdinWrite(authCtx, request)
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
		return c.stub().TaskExecStdinStatus(authCtx, request)
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
	if err := c.beginOp(); err != nil {
		return err
	}
	defer c.endOp()

	// Cancel the stream when bailing out before CloseAndRecv completes so an
	// abandoned attempt doesn't leak. A canceled stream ends without End,
	// which leaves stdin open server-side for resume.
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := c.stub().TaskExecStdinWriteStream(c.authContext(attemptCtx))
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
			return c.stub().TaskExecWait(callCtx, request)
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

const (
	stdioRetryInitialDelay = 10 * time.Millisecond
	stdioRetryDelayFactor  = 2.0
	stdioMaxRetries        = 10
)

type stdioRetry struct {
	client *taskCommandRouterClient

	delay            time.Duration
	retriesRemaining int
	didAuthRetry     bool
}

func newStdioRetry(c *taskCommandRouterClient) stdioRetry {
	return stdioRetry{
		client:           c,
		delay:            stdioRetryInitialDelay,
		retriesRemaining: stdioMaxRetries,
	}
}

func (s *stdioRetry) reset() {
	s.didAuthRetry = false
	s.delay = stdioRetryInitialDelay
	s.retriesRemaining = stdioMaxRetries
}

// recoverFrom determines whether or not to retry a failed open or read, and
// with what error if not. It waits out the backoff itself.
func (s *stdioRetry) recoverFrom(
	ctx context.Context,
	err error,
	beforeBackoff func(delay time.Duration) error,
	wrapCtxErr func(error) error,
) (retry bool, fatal error) {
	errStatus := status.Code(err)

	if errStatus == codes.Unauthenticated && !s.didAuthRetry {
		if refreshErr := s.client.refreshJwt(ctx); refreshErr != nil {
			return false, refreshErr
		}
		s.didAuthRetry = true
		return true, nil
	}
	if s.client.closed.Load() && errStatus == codes.Canceled {
		return false, ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
	}
	if _, retryable := commandRouterRetryableCodes[errStatus]; !retryable || s.retriesRemaining <= 0 {
		return false, err
	}
	if beforeBackoff != nil {
		if giveUp := beforeBackoff(s.delay); giveUp != nil {
			return false, giveUp
		}
	}

	s.client.logger.DebugContext(ctx, "Retrying stdio read with delay", "delay", s.delay, "error", err)
	select {
	case <-ctx.Done():
		ctxErr := ctx.Err()
		if wrapCtxErr != nil {
			ctxErr = wrapCtxErr(ctxErr)
		}
		return false, ctxErr
	case <-time.After(s.delay):
	}
	s.delay = time.Duration(float64(s.delay) * stdioRetryDelayFactor)
	s.retriesRemaining--
	return true, nil
}

// execStdioReader reads one exec's stdout or stderr.
//
// Nothing is pulled from the wire until Read asks for it: the stream is opened
// by the first read and each chunk is fetched by the read that wants it. There
// is no goroutine behind this, and it holds nothing beyond the chunk it is
// handing over, so a caller who reads part of the output and stops leaves
// nothing running. What grpc-go has already received is still buffered under
// it. Wrap it in a bufio.Reader for read-ahead.
type execStdioReader struct {
	chunkReader
	stdioRetry

	taskID string
	execID string
	fd     pb.TaskExecStdioFileDescriptor

	// Set only when the exec has a deadline, in which case running past it is a
	// timeout rather than an ordinary cancellation.
	deadline    time.Time
	hasDeadline bool

	stream       grpc.ServerStreamingClient[pb.TaskExecStdioReadResponse]
	streamCancel context.CancelFunc
	// The connection this stream was opened on. If the client has since dialled
	// again the stream is stale, and reopening it is not a fault.
	streamGeneration uint64
	// Where the next read resumes, so a reopened stream picks up where the last
	// one left off rather than repeating output the caller has seen.
	offset int64
}

// ExecStdioRead returns a reader over an exec's stdout or stderr.
func (c *taskCommandRouterClient) ExecStdioRead(
	ctx context.Context,
	taskID, execID string,
	fd pb.FileDescriptor,
	deadline *time.Time,
) io.ReadCloser {
	ctx, cancel := context.WithCancel(ctx)
	r := &execStdioReader{
		stdioRetry: newStdioRetry(c),
		taskID:     taskID,
		execID:     execID,
	}
	r.ctx, r.cancel = ctx, cancel
	r.fetch, r.dropStream = r.fill, r.closeStream

	switch fd {
	case pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT:
		r.fd = pb.TaskExecStdioFileDescriptor_TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT
	case pb.FileDescriptor_FILE_DESCRIPTOR_STDERR:
		r.fd = pb.TaskExecStdioFileDescriptor_TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDERR
	case pb.FileDescriptor_FILE_DESCRIPTOR_INFO, pb.FileDescriptor_FILE_DESCRIPTOR_UNSPECIFIED:
		r.err = fmt.Errorf("unsupported file descriptor: %v", fd)
	default:
		r.err = fmt.Errorf("invalid file descriptor: %v", fd)
	}

	if deadline != nil {
		r.deadline, r.hasDeadline = *deadline, true
		r.ctx, r.cancel = context.WithDeadline(ctx, *deadline)
	}
	return r
}

func (r *execStdioReader) closeStream() {
	if r.streamCancel != nil {
		r.streamCancel()
		r.streamCancel = nil
	}
	r.stream = nil
}

// fill pulls the next chunk into pending, opening or reopening the stream as it
// needs to. It returns io.EOF once the output is finished.
func (r *execStdioReader) fill() error {
	// A read in progress is what keeps the client in use. Between reads it may
	// go idle and give its connection back, taking this stream with it; the
	// next read dials again and reopens where it left off.
	if err := r.client.beginOp(); err != nil {
		return err
	}
	defer r.client.endOp()

	for {
		if err := r.ctx.Err(); err != nil {
			return r.contextErr(err)
		}

		if r.stream == nil {
			if err := r.openStream(); err != nil {
				if retry, fatal := r.recoverFrom(err); !retry {
					return fatal
				}
				continue
			}
		}

		item, err := r.stream.Recv()
		if err == io.EOF {
			return io.EOF
		}
		if err != nil {
			stale := r.streamGeneration != r.client.generation.Load()
			r.closeStream()
			if stale {
				// The connection this stream was on was given up for idleness.
				// Nothing went wrong, so reopen without spending a retry.
				continue
			}
			if retry, fatal := r.recoverFrom(err); !retry {
				return fatal
			}
			continue
		}

		r.reset()
		r.offset += int64(len(item.GetData()))
		// One chunk per message, and fill only runs with pending empty, so the
		// chunk replaces it. The log reader concatenates instead because a
		// batch there carries several items.
		r.pending = item.GetData()
		return nil
	}
}

func (r *execStdioReader) openStream() error {
	generation := r.client.generation.Load()
	streamCtx, cancel := context.WithCancel(r.ctx)
	stream, err := r.client.stub().TaskExecStdioRead(
		r.client.authContext(streamCtx),
		pb.TaskExecStdioReadRequest_builder{
			TaskId:         r.taskID,
			ExecId:         r.execID,
			Offset:         uint64(r.offset),
			FileDescriptor: r.fd,
		}.Build(),
	)
	if err != nil {
		cancel()
		return err
	}
	r.stream, r.streamCancel = stream, cancel
	r.streamGeneration = generation
	return nil
}

// recoverFrom says whether a failed open or read is worth another go, and with
// what error if it is not. It waits out the backoff itself.
func (r *execStdioReader) recoverFrom(err error) (retry bool, fatal error) {
	return r.stdioRetry.recoverFrom(r.ctx, err, func(delay time.Duration) error {
		// Sleeping past the exec's deadline is a timeout, not a retry.
		if r.hasDeadline && time.Until(r.deadline) <= delay {
			return r.timeoutErr()
		}
		return nil
	}, r.contextErr)
}

func (r *execStdioReader) contextErr(err error) error {
	if r.hasDeadline && errors.Is(err, context.DeadlineExceeded) {
		return r.timeoutErr()
	}
	return err
}

func (r *execStdioReader) timeoutErr() error {
	return ExecTimeoutError{
		Exception: fmt.Sprintf("deadline exceeded while streaming stdio for exec %s", r.execID),
	}
}

// sandboxStdioReader reads a V2 Sandbox's stdout or stderr.
type sandboxStdioReader struct {
	chunkReader
	stdioRetry

	taskID string
	fd     pb.SandboxStdioFileDescriptor

	stream       grpc.ServerStreamingClient[pb.SandboxStdioReadV2Response]
	streamCancel context.CancelFunc
	// The connection this stream was opened on.
	streamGeneration uint64
	// Where the next read resumes, so a reopened stream picks up where the last
	// one left off rather than repeating output the caller has seen.
	offset int64
}

// SandboxStdioReadV2 returns a reader over a V2 Sandbox's stdout or stderr.
func (c *taskCommandRouterClient) SandboxStdioReadV2(
	ctx context.Context,
	taskID string,
	fd pb.FileDescriptor,
) io.ReadCloser {
	ctx, cancel := context.WithCancel(ctx)
	r := &sandboxStdioReader{
		stdioRetry: newStdioRetry(c),
		taskID:     taskID,
	}
	r.ctx, r.cancel = ctx, cancel
	r.fetch, r.dropStream = r.fill, r.closeStream

	switch fd {
	case pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT:
		r.fd = pb.SandboxStdioFileDescriptor_SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT
	case pb.FileDescriptor_FILE_DESCRIPTOR_STDERR:
		r.fd = pb.SandboxStdioFileDescriptor_SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR
	case pb.FileDescriptor_FILE_DESCRIPTOR_INFO, pb.FileDescriptor_FILE_DESCRIPTOR_UNSPECIFIED:
		r.err = fmt.Errorf("unsupported file descriptor: %v", fd)
	default:
		r.err = fmt.Errorf("invalid file descriptor: %v", fd)
	}
	return r
}

func (r *sandboxStdioReader) closeStream() {
	if r.streamCancel != nil {
		r.streamCancel()
		r.streamCancel = nil
	}
	r.stream = nil
}

// fill pulls the next chunk into pending, opening or reopening the stream as it
// needs to. It returns io.EOF once the output is finished.
func (r *sandboxStdioReader) fill() error {
	if err := r.client.beginOp(); err != nil {
		return err
	}
	defer r.client.endOp()

	for {
		if err := r.ctx.Err(); err != nil {
			return err
		}

		opened := false
		if r.stream == nil {
			if err := r.openStream(); err != nil {
				if retry, fatal := r.recoverFrom(err); !retry {
					return fatal
				}
				continue
			}
			opened = true
		}

		item, err := r.stream.Recv()
		if err == io.EOF {
			return io.EOF
		}
		if err != nil {
			stale := r.streamGeneration != r.client.generation.Load()
			r.closeStream()
			if stale {
				// The connection this stream was on was given up for idleness.
				// Reopen without spending a retry.
				continue
			}
			if retry, fatal := r.recoverFrom(err); !retry {
				return fatal
			}
			continue
		}

		data := item.GetData()
		if len(data) == 0 {
			return fmt.Errorf("received empty message streaming stdio from Sandbox %s", r.client.sandboxID)
		}
		if opened {
			if dropped := int64(item.GetStartingOffset()) - r.offset; dropped > 0 {
				r.client.logger.WarnContext(r.ctx,
					"V2 Sandbox stdio: dropped bytes. Only the most recent portion of output is retained",
					"sandbox_id", r.client.sandboxID, "dropped_bytes", dropped)
			}
			r.offset = int64(item.GetStartingOffset())
		}

		r.reset()
		r.offset += int64(len(data))
		r.pending = data
		return nil
	}
}

func (r *sandboxStdioReader) openStream() error {
	generation := r.client.generation.Load()
	streamCtx, cancel := context.WithCancel(r.ctx)
	stream, err := r.client.stub().SandboxStdioReadV2(
		r.client.authContext(streamCtx),
		pb.SandboxStdioReadV2Request_builder{
			TaskId:         r.taskID,
			Offset:         uint64(r.offset),
			FileDescriptor: r.fd,
		}.Build(),
	)
	if err != nil {
		cancel()
		return err
	}
	r.stream, r.streamCancel = stream, cancel
	r.streamGeneration = generation
	return nil
}

func (r *sandboxStdioReader) recoverFrom(err error) (retry bool, fatal error) {
	return r.stdioRetry.recoverFrom(r.ctx, err, nil, nil)
}

// SandboxStdinWriteV2 writes data to stdin of a V2 Sandbox's entrypoint process.
func (c *taskCommandRouterClient) SandboxStdinWriteV2(ctx context.Context, taskID string, offset uint64, data []byte, eof bool) error {
	request := pb.SandboxStdinWriteV2Request_builder{
		TaskId: taskID,
		Offset: offset,
		Data:   data,
		Eof:    eof,
	}.Build()

	_, err := callCommandRouterUnary(ctx, c, func(authCtx context.Context) (*pb.SandboxStdinWriteV2Response, error) {
		return c.stub().SandboxStdinWriteV2(authCtx, request)
	})
	return err
}
