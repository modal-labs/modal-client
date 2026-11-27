package modal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Lifetime limits of each session of a persistent Sandbox. A session that hits
// one is paused.
const (
	persistentSandboxSessionTimeout     = 60 * time.Minute
	persistentSandboxSessionIdleTimeout = 5 * time.Minute
)

// How long a session lets running commands finish before it stops, by
// default and at most.
const (
	persistentSandboxDrainTimeout    = time.Minute
	persistentSandboxMaxDrainTimeout = 10 * time.Minute
)

// The experimental option that sets a persistent Sandbox's drain timeout, in
// whole seconds.
const drainTimeoutSecsOption = "drain_timeout_secs"

// persistentDrainTimeout returns the drain timeout the "drain_timeout_secs"
// experimental option sets, or the default when it is unset.
func persistentDrainTimeout(options map[string]any) (time.Duration, error) {
	value, set := options[drainTimeoutSecsOption]
	if !set {
		return persistentSandboxDrainTimeout, nil
	}
	secs, ok := value.(int)
	if !ok {
		return 0, InvalidError{Exception: fmt.Sprintf(`Experimental option "drain_timeout_secs" must be an int, got %T.`, value)}
	}
	drainTimeout := time.Duration(secs) * time.Second
	if drainTimeout <= 0 || drainTimeout > persistentSandboxMaxDrainTimeout {
		return 0, InvalidError{Exception: fmt.Sprintf(`Experimental option "drain_timeout_secs" must be between 1 and %d, got %d.`, int(persistentSandboxMaxDrainTimeout/time.Second), secs)}
	}
	return drainTimeout, nil
}

// Resuming is unavailable while the previous session is still saving its
// state or another resume is in progress, and pausing is while a session is
// starting, so both are retried this long at this interval.
const (
	persistentBusyRetryTimeout  = 120 * time.Second
	persistentBusyRetryInterval = time.Second
)

// retryWhileBusy calls call until it succeeds, retrying retryable errors. Only
// for calls that are idempotent.
func retryWhileBusy[T any](ctx context.Context, call func() (T, error)) (T, error) {
	deadline := time.Now().Add(persistentBusyRetryTimeout)
	for {
		resp, err := call()
		if err == nil || !isRetryableGrpc(err) || time.Now().After(deadline) {
			return resp, err
		}
		select {
		case <-ctx.Done():
			var zero T
			return zero, ctx.Err()
		case <-time.After(persistentBusyRetryInterval):
		}
	}
}

// How often wait checks whether the persistent Sandbox has been terminated.
const persistentWaitPollInterval = time.Second

var persistentTerminatedResult = pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_TERMINATED}.Build()

// isNonRecoverableError reports errors caused by the caller rather than by the
// session, which a new session can't fix.
func isNonRecoverableError(err error) bool {
	var invalid InvalidError
	var closed ClientClosedError
	return errors.As(err, &invalid) || errors.As(err, &closed)
}

// rejectedByShuttingDownSession reports whether the session was shutting down
// and never started the failed request.
func rejectedByShuttingDownSession(err error) bool {
	st, ok := status.FromError(err)
	if !ok {
		return false
	}
	for _, detail := range st.Details() {
		if _, ok := detail.(*pb.TaskShuttingDown); ok {
			return true
		}
	}
	return false
}

func canReportPaused(err error) bool {
	return rejectedByShuttingDownSession(err) || !isNonRecoverableError(err)
}

// execNotSentError marks a failure to reach a session's command router, which
// comes before the command is sent to it.
type execNotSentError struct{ err error }

func (e execNotSentError) Error() string { return e.err.Error() }
func (e execNotSentError) Unwrap() error { return e.err }

// execNeverRan reports whether a failed exec certainly didn't run its command,
// so it can be started on another session.
func execNeverRan(err error) bool {
	var notSent execNotSentError
	if errors.As(err, &notSent) {
		// Reaching the router takes no input from the caller, so only a
		// closed client is their doing.
		var closed ClientClosedError
		return !errors.As(notSent.err, &closed)
	}
	if rejectedByShuttingDownSession(err) {
		return true
	}
	// A NotFound status means the session was gone before the command could
	// reach it. Any other failure, including a dropped connection, may come
	// after the command started.
	return !isNonRecoverableError(err) && status.Code(err) == codes.NotFound
}

// sandboxImplPersistent backs a persistent Sandbox, whose object ID is the ID
// of the persistent Sandbox.
//
// A stopped session requires an explicit resume before running more work.
type sandboxImplPersistent struct {
	client *Client
	id     string // the persistent Sandbox ID

	mu         sync.Mutex // protects session, terminated, closed
	session    *sandboxImplV2
	terminated bool
	closed     bool
	// sessionLookupMu serializes the initial session lookup with resume.
	sessionLookupMu sync.Mutex
}

// sandboxImplPersistentParams is optional metadata already known about a
// persistent Sandbox. Anything left out is looked up on first use.
type sandboxImplPersistentParams struct {
	// sessionID is the ID of the running session.
	sessionID string
}

func newSandboxImplPersistent(client *Client, id string, params sandboxImplPersistentParams) *sandboxImplPersistent {
	p := &sandboxImplPersistent{client: client, id: id}
	if params.sessionID != "" {
		p.session = newSandboxImplV2(client, params.sessionID, sandboxImplParams{})
	}
	return p
}

func (p *sandboxImplPersistent) ensureSession(ctx context.Context) (*sandboxImplV2, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	session, err := p.sessionForReads(ctx)
	if err != nil {
		return nil, err
	}
	if err := p.checkNotTerminated(); err != nil {
		return nil, err
	}
	if session == nil {
		return nil, persistentSessionPaused()
	}
	return session, nil
}

// createSandboxPersistent creates a persistent Sandbox and returns the
// implementation for its first session.
func createSandboxPersistent(ctx context.Context, client *Client, app *App, image *Image, params *SandboxCreateParams) (sandboxImpl, error) {
	if params.Timeout != 0 || params.IdleTimeout != 0 {
		return nil, InvalidError{Exception: "Persistent Sandboxes manage their own session timeouts and don't support Timeout or IdleTimeout."}
	}
	if params.Name != "" {
		return nil, InvalidError{Exception: "Named persistent Sandboxes are not supported yet."}
	}
	if params.GPU != "" || params.PTY {
		return nil, InvalidError{Exception: "GPUs and PTYs are not supported by persistent Sandboxes yet."}
	}
	if params.ExperimentalEnableSnapshot {
		return nil, InvalidError{Exception: "Memory snapshots are not supported by persistent Sandboxes."}
	}
	if len(params.Env) > 0 || len(params.Tags) > 0 {
		return nil, InvalidError{Exception: "Env and Tags are not supported by persistent Sandboxes yet."}
	}
	drainTimeout, err := persistentDrainTimeout(params.ExperimentalOptions)
	if err != nil {
		return nil, err
	}

	image, err = image.Build(ctx, app, nil)
	if err != nil {
		return nil, err
	}

	resolvable := resolvableSecrets(params.Secrets)
	if err := hydrateSandboxSecrets(ctx, client, append(append(append([]*Secret{}, resolvable...), params.ExperimentalOutboundPolicy.secrets()...), params.OutboundPolicy.secrets()...), params.CloudBucketMounts); err != nil {
		return nil, err
	}
	// Every session is created from the same definition, so credentials must
	// live in server-side Secrets rather than inline in the request.
	sources, err := secretSources(params.Secrets, nil)
	if err != nil {
		return nil, err
	}
	for _, source := range sources {
		if source.HasEnv() {
			return nil, InvalidError{Exception: "Ephemeral credentials are not supported by persistent Sandboxes yet."}
		}
	}

	definitionParams := *params
	definitionParams.Secrets = resolvable
	definitionParams.Timeout = persistentSandboxSessionTimeout
	definitionParams.IdleTimeout = persistentSandboxSessionIdleTimeout
	definitionParams.drainTimeout = drainTimeout
	v2Req, err := buildSandboxCreateV2RequestProto(app.AppID, image.ImageID, definitionParams, sources)
	if err != nil {
		return nil, err
	}
	warnDeprecatedNetworkParams(ctx, client.logger, params)

	resp, err := client.cpClient.SxSandboxCreate(ctx, pb.SxSandboxCreateRequest_builder{
		AppId:      v2Req.GetAppId(),
		Definition: v2Req.GetDefinition(),
	}.Build())
	if err != nil {
		return nil, err
	}
	client.logger.DebugContext(ctx, "Created persistent Sandbox",
		"persistent_sandbox_id", resp.GetSxId(),
		"sandbox_id", resp.GetSandboxId())
	return newSandboxImplPersistent(client, resp.GetSxId(), sandboxImplPersistentParams{sessionID: resp.GetSandboxId()}), nil
}

func (p *sandboxImplPersistent) sandboxID() string    { return p.id }
func (p *sandboxImplPersistent) modalClient() *Client { return p.client }

// currentSession returns the session operations are currently forwarded to.
func (p *sandboxImplPersistent) currentSession() *sandboxImplV2 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.session
}

func (p *sandboxImplPersistent) isTerminated() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.terminated
}

// Session lookup and resume

func persistentSessionPaused() SandboxPaused {
	return SandboxPaused{Exception: "The current Sandbox session has stopped or is shutting down. Call ExperimentalResume and retry."}
}

// run reports a session that stopped accepting work if the operation can
// safely be retried.
func run[T any](
	ctx context.Context,
	p *sandboxImplPersistent,
	canReport func(error) bool,
	op func(session *sandboxImplV2) (T, error),
) (T, error) {
	session, sessionErr := p.ensureSession(ctx)
	if sessionErr != nil {
		var zero T
		return zero, sessionErr
	}
	result, err := op(session)
	if err != nil && ctx.Err() == nil && !p.isTerminated() && canReport(err) && p.sessionHasStopped(ctx, session, err) {
		return result, persistentSessionPaused()
	}
	return result, err
}

func (p *sandboxImplPersistent) sessionHasStopped(ctx context.Context, session *sandboxImplV2, opErr error) bool {
	sessionID, err := p.getCurrentSessionID(ctx)
	if err != nil || p.isTerminated() {
		return false
	}
	if sessionID != session.id || rejectedByShuttingDownSession(opErr) {
		return true
	}
	result, err := session.poll(ctx)
	return result != nil || status.Code(err) == codes.NotFound
}

func runErr(ctx context.Context, p *sandboxImplPersistent, op func(session *sandboxImplV2) error) error {
	_, err := run(ctx, p, canReportPaused, func(session *sandboxImplV2) (struct{}, error) {
		return struct{}{}, op(session)
	})
	return err
}

// getCurrentSessionID looks up the most recent session, which may have
// finished already. It returns "" if there has never been one. Only
// termination is final: whether the session is still running is learned from
// the session itself.
func (p *sandboxImplPersistent) getCurrentSessionID(ctx context.Context) (string, error) {
	resp, err := p.client.cpClient.SxSandboxGet(ctx, pb.SxSandboxGetRequest_builder{
		SxId: &p.id,
	}.Build())
	if err != nil {
		return "", err
	}
	p.mu.Lock()
	p.terminated = p.terminated || resp.HasTerminatedAt()
	terminated := p.terminated
	p.mu.Unlock()
	if terminated && resp.HasCurrentSandboxId() {
		// The last session, which holds the exit snapshot and logs.
		p.setSession(ctx, resp.GetCurrentSandboxId())
	}
	return resp.GetCurrentSandboxId(), nil
}

// resumeSession resumes the persistent Sandbox, returning the ID of its
// running session.
func (p *sandboxImplPersistent) resumeSession(ctx context.Context) (string, error) {
	req := pb.SxSandboxResumeRequest_builder{SxId: p.id}.Build()
	resp, err := retryWhileBusy(ctx, func() (*pb.SxSandboxResumeResponse, error) {
		return p.client.cpClient.SxSandboxResume(ctx, req)
	})
	if err != nil {
		return "", err
	}
	return resp.GetSandboxId(), nil
}

// setSession makes sessionID the current session and closes the session it
// replaces.
//
// Every session switch goes through here, so a replaced session never keeps a
// connection open, and a detached Sandbox only ever holds closed sessions.
func (p *sandboxImplPersistent) setSession(ctx context.Context, sessionID string) *sandboxImplV2 {
	p.mu.Lock()
	previous := p.session
	if previous != nil && previous.id == sessionID {
		p.mu.Unlock()
		return previous
	}
	session := newSandboxImplV2(p.client, sessionID, sandboxImplParams{})
	p.session = session
	closed := p.closed
	p.mu.Unlock()
	p.client.logger.DebugContext(ctx, "Switching persistent Sandbox to session",
		"persistent_sandbox_id", p.id, "sandbox_id", sessionID)
	if previous != nil {
		previous.close()
	}
	if closed {
		session.close()
	}
	return session
}

// adoptCurrentSession switches to the most recent session without resuming.
// It returns nil if there has never been one.
func (p *sandboxImplPersistent) adoptCurrentSession(ctx context.Context) (*sandboxImplV2, error) {
	sessionID, err := p.getCurrentSessionID(ctx)
	if err != nil || sessionID == "" {
		return nil, err
	}
	return p.setSession(ctx, sessionID), nil
}

// sessionForReads returns the last known session for reading its logs or
// metadata, without resuming. It returns nil if there is none.
func (p *sandboxImplPersistent) sessionForReads(ctx context.Context) (*sandboxImplV2, error) {
	if session := p.currentSession(); session != nil || p.isTerminated() {
		return session, nil
	}
	p.sessionLookupMu.Lock()
	defer p.sessionLookupMu.Unlock()
	if session := p.currentSession(); session != nil || p.isTerminated() {
		return session, nil
	}
	return p.adoptCurrentSession(ctx)
}

func (p *sandboxImplPersistent) checkOpen() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
	}
	return nil
}

func (p *sandboxImplPersistent) checkNotTerminated() error {
	if p.isTerminated() {
		return InvalidError{Exception: fmt.Sprintf("Persistent Sandbox %s has been terminated.", p.id)}
	}
	return nil
}

// Operations

func (p *sandboxImplPersistent) stdin() io.WriteCloser {
	// Streams connect to whichever session is current when they first connect.
	return &sbStdinV2{getCommandRouter: p.getCommandRouter}
}

func (p *sandboxImplPersistent) outputStream(fd pb.FileDescriptor) io.ReadCloser {
	ctx, cancel := context.WithCancel(context.Background())
	return &lazyStreamReader{
		readCheck: func() error {
			if err := p.checkOpen(); err != nil {
				return err
			}
			if session := p.currentSession(); session != nil {
				return session.commandRouter.ensureOpen()
			}
			return nil
		},
		cancel: cancel,
		initFunc: func() io.ReadCloser {
			return outputStreamSbV2(ctx, p.getCommandRouter, fd)
		},
	}
}

func (p *sandboxImplPersistent) getTags(ctx context.Context) (map[string]string, error) {
	return nil, InvalidError{Exception: "Tags aren't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) setTags(ctx context.Context, tags map[string]string) error {
	return InvalidError{Exception: "Tags aren't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) setName(ctx context.Context, name string) error {
	return InvalidError{Exception: "The name of a persistent Sandbox can't be changed."}
}

// createConnectToken addresses the persistent Sandbox, not a session, so its
// tokens keep working across sessions and are created without resuming it.
func (p *sandboxImplPersistent) createConnectToken(ctx context.Context, port uint32, userMetadata string) (*SandboxCreateConnectCredentials, error) {
	resp, err := p.client.cpClient.SxSandboxCreateConnectToken(ctx, pb.SxSandboxCreateConnectTokenRequest_builder{
		SxId:         p.id,
		UserMetadata: userMetadata,
		Port:         &port,
	}.Build())
	if err != nil {
		return nil, err
	}
	return &SandboxCreateConnectCredentials{URL: resp.GetUrl(), Token: resp.GetToken()}, nil
}

func (p *sandboxImplPersistent) terminate(ctx context.Context) error {
	resp, err := p.client.cpClient.SxSandboxTerminate(ctx, pb.SxSandboxTerminateRequest_builder{
		SxId: p.id,
	}.Build())
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.terminated = true
	p.mu.Unlock()
	// The last session, which holds the exit snapshot and logs.
	if resp.HasLastSessionSandboxId() {
		p.setSession(ctx, resp.GetLastSessionSandboxId())
	}
	return nil
}

// pause stops the current session, which saves its state for the next resume,
// and waits for it to stop.
//
// It does nothing if the Sandbox is already paused. A paused Sandbox is never
// resumed in order to pause it.
func (p *sandboxImplPersistent) pause(ctx context.Context) error {
	if err := p.checkNotTerminated(); err != nil {
		return err
	}
	req := pb.SxSandboxPauseRequest_builder{SxId: p.id}.Build()
	resp, err := retryWhileBusy(ctx, func() (*pb.SxSandboxPauseResponse, error) {
		return p.client.cpClient.SxSandboxPause(ctx, req)
	})
	if err != nil || !resp.HasSandboxId() {
		return err
	}
	// Wait on the session the pause stops, even if another handle has resumed
	// the Sandbox since.
	_, err = p.setSession(ctx, resp.GetSandboxId()).wait(ctx)
	return err
}

// resume starts a session from the latest saved state, unless one is already
// running.
func (p *sandboxImplPersistent) resume(ctx context.Context) error {
	p.sessionLookupMu.Lock()
	defer p.sessionLookupMu.Unlock()
	if err := p.checkNotTerminated(); err != nil {
		return err
	}
	sessionID, err := p.resumeSession(ctx)
	if err != nil {
		return err
	}
	p.setSession(ctx, sessionID)
	return nil
}

func (p *sandboxImplPersistent) close() {
	p.mu.Lock()
	p.closed = true
	p.mu.Unlock()
	if session := p.currentSession(); session != nil {
		session.close()
	}
}

func (p *sandboxImplPersistent) wait(ctx context.Context) (int, error) {
	for {
		result, err := p.poll(ctx)
		if err != nil {
			return 0, err
		}
		if result != nil {
			return *result, nil
		}
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(persistentWaitPollInterval):
		}
	}
}

func (p *sandboxImplPersistent) poll(ctx context.Context) (*int, error) {
	if !p.isTerminated() {
		var err error
		if p.currentSession() == nil {
			_, err = p.sessionForReads(ctx)
		} else {
			_, err = p.getCurrentSessionID(ctx)
		}
		if err != nil {
			return nil, err
		}
	}
	if p.isTerminated() {
		return getReturnCode(persistentTerminatedResult), nil
	}
	return nil, nil
}

func (p *sandboxImplPersistent) waitForScheduling(ctx context.Context, timeout time.Duration) error {
	session, err := p.ensureSession(ctx)
	if err != nil {
		return err
	}
	return session.waitForScheduling(ctx, timeout)
}

// tunnels are addressed by the persistent Sandbox, not a session, so they are
// served without a running session and never go stale across sessions.
func (p *sandboxImplPersistent) tunnels(ctx context.Context, _ time.Duration) (map[int]*Tunnel, error) {
	resp, err := p.client.cpClient.SxSandboxTunnelsList(ctx, pb.SxSandboxTunnelsListRequest_builder{
		SxId: p.id,
	}.Build())
	if err != nil {
		return nil, err
	}
	return tunnelsFromProtos(resp.GetTunnels()), nil
}

// getAppID and resolveTaskIDForLogs read from the last known session, so
// reading logs never resumes a paused Sandbox.
func (p *sandboxImplPersistent) getAppID(ctx context.Context) (string, error) {
	session, err := p.sessionForLogs(ctx)
	if err != nil {
		return "", err
	}
	return session.getAppID(ctx)
}

func (p *sandboxImplPersistent) resolveTaskIDForLogs(ctx context.Context) (string, error) {
	session, err := p.sessionForLogs(ctx)
	if err != nil {
		return "", err
	}
	return session.resolveTaskIDForLogs(ctx)
}

func (p *sandboxImplPersistent) sessionForLogs(ctx context.Context) (*sandboxImplV2, error) {
	session, err := p.sessionForReads(ctx)
	if err != nil {
		return nil, err
	}
	if session == nil {
		return nil, InvalidError{Exception: fmt.Sprintf("Persistent Sandbox %s has no session to read logs from.", p.id)}
	}
	return session, nil
}

// getExitSnapshot waits for the persistent Sandbox to be terminated. Pausing
// is not an exit: only termination produces its exit snapshot.
func (p *sandboxImplPersistent) getExitSnapshot(ctx context.Context, params *SandboxExperimentalGetExitSnapshotParams) (*Image, error) {
	var deadline time.Time
	hasDeadline := params != nil && params.Timeout != nil
	if hasDeadline {
		deadline = time.Now().Add(*params.Timeout)
	}
	for {
		result, err := p.poll(ctx)
		if err != nil {
			return nil, err
		}
		if result != nil {
			break
		}
		if hasDeadline && time.Now().Add(persistentWaitPollInterval).After(deadline) {
			return nil, TimeoutError{Exception: fmt.Sprintf("timed out waiting for exit snapshot for Sandbox %s", p.id)}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(persistentWaitPollInterval):
		}
	}
	session := p.currentSession()
	if session == nil {
		return nil, InvalidError{Exception: fmt.Sprintf("Persistent Sandbox %s has no exit snapshot: it was terminated without a session.", p.id)}
	}
	sessionParams := &SandboxExperimentalGetExitSnapshotParams{}
	if hasDeadline {
		remaining := max(time.Until(deadline), 0)
		sessionParams.Timeout = &remaining
	}
	return session.getExitSnapshot(ctx, sessionParams)
}

func (p *sandboxImplPersistent) snapshotMemory(ctx context.Context) (*SandboxSnapshot, error) {
	return nil, InvalidError{Exception: "Memory snapshots aren't supported for persistent Sandboxes."}
}

func (p *sandboxImplPersistent) getCommandRouter(ctx context.Context) (string, *taskCommandRouterClient, error) {
	if err := p.checkOpen(); err != nil {
		return "", nil, err
	}
	type router struct {
		taskID string
		client *taskCommandRouterClient
	}
	r, err := run(ctx, p, canReportPaused, func(s *sandboxImplV2) (router, error) {
		taskID, client, err := s.getCommandRouter(ctx)
		return router{taskID, client}, err
	})
	return r.taskID, r.client, err
}

func (p *sandboxImplPersistent) waitUntilReady(ctx context.Context, timeout time.Duration) error {
	return runErr(ctx, p, func(s *sandboxImplV2) error { return s.waitUntilReady(ctx, timeout) })
}

func (p *sandboxImplPersistent) updateNetworkPolicy(ctx context.Context, params SandboxUpdateNetworkPolicyParams) error {
	return InvalidError{Exception: "Updating network access isn't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) updateOutboundPolicy(ctx context.Context, policy *ExperimentalOutboundPolicy) error {
	return InvalidError{Exception: "Updating the outbound policy isn't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) setOutboundPolicy(ctx context.Context, req *pb.SandboxSetOutboundPolicyRequest, opts ...grpc.CallOption) error {
	return InvalidError{Exception: "Setting the outbound policy isn't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) snapshotFilesystem(ctx context.Context, timeout, ttl time.Duration, containerID string) (*Image, error) {
	return run(ctx, p, canReportPaused, func(s *sandboxImplV2) (*Image, error) {
		return s.snapshotFilesystem(ctx, timeout, ttl, containerID)
	})
}

func (p *sandboxImplPersistent) snapshotDirectory(ctx context.Context, path string, timeout, ttl time.Duration, params *SandboxSnapshotDirectoryParams) (*Image, error) {
	return run(ctx, p, canReportPaused, func(s *sandboxImplV2) (*Image, error) {
		return s.snapshotDirectory(ctx, path, timeout, ttl, params)
	})
}

func (p *sandboxImplPersistent) mountImage(ctx context.Context, path string, image *Image, params *SandboxMountImageParams) error {
	return InvalidError{Exception: "Mounting images isn't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) unmountImage(ctx context.Context, path string) error {
	return InvalidError{Exception: "Mounting images isn't supported for persistent Sandboxes yet."}
}

func (p *sandboxImplPersistent) reloadVolumes(ctx context.Context, containerID string, timeout time.Duration) error {
	return runErr(ctx, p, func(s *sandboxImplV2) error { return s.reloadVolumes(ctx, containerID, timeout) })
}

func (p *sandboxImplPersistent) exec(ctx context.Context, command []string, params *SandboxExecParams, containerID string) (*ContainerProcess, error) {
	process, err := run(ctx, p, execNeverRan, func(s *sandboxImplV2) (*ContainerProcess, error) {
		if _, _, err := s.getCommandRouter(ctx); err != nil {
			return nil, execNotSentError{err}
		}
		return s.exec(ctx, command, params, containerID)
	})
	var notSent execNotSentError
	if errors.As(err, &notSent) {
		err = notSent.err
	}
	return process, err
}

func (p *sandboxImplPersistent) validateSidecarCreate(params *SidecarCreateParams) error {
	return newSandboxImplV2(p.client, "", sandboxImplParams{}).validateSidecarCreate(params)
}

func (p *sandboxImplPersistent) createSidecar(ctx context.Context, in sidecarCreateInputs) (sidecarCreateResult, error) {
	return run(ctx, p, canReportPaused, func(s *sandboxImplV2) (sidecarCreateResult, error) { return s.createSidecar(ctx, in) })
}
