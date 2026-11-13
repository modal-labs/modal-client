package modal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Lifetime limits of each session of a persistent Sandbox. A session that hits
// one is paused, and resumed on next use.
const (
	persistentSandboxSessionTimeout     = 60 * time.Minute
	persistentSandboxSessionIdleTimeout = 5 * time.Minute
)

// How long to retry resuming while the previous session is still saving its state.
const (
	persistentResumeRetryTimeout  = 120 * time.Second
	persistentResumeRetryInterval = time.Second
)

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

// sandboxImplPersistent backs a persistent Sandbox, whose object ID is the ID
// of its first session.
//
// A persistent Sandbox runs as a sequence of regular V2 Sandboxes, its
// sessions, each started from the saved state of the previous one. Operations
// are forwarded to the current session. When one fails and the Sandbox hasn't
// been terminated through this handle, the running session is looked up,
// resuming the Sandbox from its latest saved state if there is none, and the
// operation is retried once on the new session.
type sandboxImplPersistent struct {
	client              *Client
	id                  string
	persistentSandboxID string

	mu         sync.Mutex // protects session, terminated
	session    *sandboxImplV2
	terminated bool
	// replaceMu serializes session replacements, so concurrent failures on
	// the same session trigger a single lookup/resume.
	replaceMu sync.Mutex
}

func newSandboxImplPersistent(client *Client, sandboxID, persistentSandboxID string) *sandboxImplPersistent {
	return &sandboxImplPersistent{
		client:              client,
		id:                  sandboxID,
		persistentSandboxID: persistentSandboxID,
		session:             newSandboxImplV2(client, sandboxID, sandboxImplParams{}),
	}
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

	image, err := image.Build(ctx, app, nil)
	if err != nil {
		return nil, err
	}

	resolvable := resolvableSecrets(params.Secrets)
	if err := hydrateSandboxSecrets(ctx, client, append(append([]*Secret{}, resolvable...), params.ExperimentalOutboundPolicy.secrets()...), params.CloudBucketMounts); err != nil {
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
	v2Req, err := buildSandboxCreateV2RequestProto(app.AppID, image.ImageID, definitionParams, sources)
	if err != nil {
		return nil, err
	}

	resp, err := client.cpClient.PersistentSandboxCreate(ctx, pb.PersistentSandboxCreateRequest_builder{
		AppId:      v2Req.GetAppId(),
		Definition: v2Req.GetDefinition(),
	}.Build())
	if err != nil {
		return nil, err
	}
	client.logger.DebugContext(ctx, "Created persistent Sandbox",
		"persistent_sandbox_id", resp.GetPersistentSandboxId(),
		"sandbox_id", resp.GetSandboxId())
	return newSandboxImplPersistent(client, resp.GetSandboxId(), resp.GetPersistentSandboxId()), nil
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

// Session replacement

// run runs op on the session, switching to the running session and retrying
// once if it fails.
func run[T any](ctx context.Context, p *sandboxImplPersistent, op func(session *sandboxImplV2) (T, error)) (T, error) {
	session := p.currentSession()
	result, err := op(session)
	if err == nil {
		return result, nil
	}
	// A terminated persistent Sandbox is stopped for good, so it isn't resumed on use.
	if isNonRecoverableError(err) || ctx.Err() != nil || p.isTerminated() {
		return result, err
	}
	replacement, replaceErr := p.replaceSession(ctx, session)
	if replaceErr != nil {
		return result, replaceErr
	}
	if replacement == nil {
		return result, err
	}
	p.client.logger.DebugContext(ctx, "Retrying operation on persistent Sandbox",
		"persistent_sandbox_id", p.persistentSandboxID, "error", err)
	return op(replacement)
}

func runErr(ctx context.Context, p *sandboxImplPersistent, op func(session *sandboxImplV2) error) error {
	_, err := run(ctx, p, func(session *sandboxImplV2) (struct{}, error) {
		return struct{}{}, op(session)
	})
	return err
}

func (p *sandboxImplPersistent) getRunningSessionID(ctx context.Context) (string, error) {
	resp, err := p.client.cpClient.PersistentSandboxGet(ctx, pb.PersistentSandboxGetRequest_builder{
		PersistentSandboxId: &p.persistentSandboxID,
	}.Build())
	if err != nil {
		return "", err
	}
	return resp.GetSandboxId(), nil
}

// resumeSession resumes the persistent Sandbox, returning the ID of its
// running session.
func (p *sandboxImplPersistent) resumeSession(ctx context.Context) (string, error) {
	req := pb.PersistentSandboxResumeRequest_builder{PersistentSandboxId: p.persistentSandboxID}.Build()
	deadline := time.Now().Add(persistentResumeRetryTimeout)
	for {
		resp, err := p.client.cpClient.PersistentSandboxResume(ctx, req)
		if err == nil {
			return resp.GetSandboxId(), nil
		}
		// Unavailable means the previous session is still pausing or another
		// resume is in progress.
		if status.Code(err) != codes.Unavailable || time.Now().After(deadline) {
			return "", err
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(persistentResumeRetryInterval):
		}
	}
}

// replaceSession switches from the failed session, or returns nil if it is
// still the running session.
func (p *sandboxImplPersistent) replaceSession(ctx context.Context, failed *sandboxImplV2) (*sandboxImplV2, error) {
	p.replaceMu.Lock()
	defer p.replaceMu.Unlock()

	if current := p.currentSession(); current != failed {
		// A concurrent operation has already replaced the session.
		return current, nil
	}
	sessionID, err := p.getRunningSessionID(ctx)
	if err != nil {
		return nil, err
	}
	if sessionID == failed.id {
		return nil, nil
	}
	if sessionID == "" {
		sessionID, err = p.resumeSession(ctx)
		if err != nil {
			return nil, err
		}
	}
	p.client.logger.DebugContext(ctx, "Switching persistent Sandbox to session",
		"persistent_sandbox_id", p.persistentSandboxID, "sandbox_id", sessionID)
	replacement := newSandboxImplV2(p.client, sessionID, sandboxImplParams{})
	p.mu.Lock()
	p.session = replacement
	p.mu.Unlock()
	failed.close()
	return replacement, nil
}

func (p *sandboxImplPersistent) checkNotTerminated() error {
	if p.isTerminated() {
		return InvalidError{Exception: fmt.Sprintf("Persistent Sandbox %s has been terminated.", p.persistentSandboxID)}
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
		readCheck: func() error { return p.currentSession().commandRouter.ensureOpen() },
		cancel:    cancel,
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

func (p *sandboxImplPersistent) createConnectToken(ctx context.Context, port uint32, userMetadata string) (*SandboxCreateConnectCredentials, error) {
	return run(ctx, p, func(s *sandboxImplV2) (*SandboxCreateConnectCredentials, error) {
		return s.createConnectToken(ctx, port, userMetadata)
	})
}

func (p *sandboxImplPersistent) terminate(ctx context.Context) error {
	_, err := p.client.cpClient.PersistentSandboxTerminate(ctx, pb.PersistentSandboxTerminateRequest_builder{
		PersistentSandboxId: p.persistentSandboxID,
	}.Build())
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.terminated = true
	p.mu.Unlock()
	return nil
}

// pause stops the current session, which saves its state for the next resume.
func (p *sandboxImplPersistent) pause(ctx context.Context) error {
	if err := p.checkNotTerminated(); err != nil {
		return err
	}
	session := p.currentSession()
	if err := session.terminate(ctx); err != nil {
		return err
	}
	_, err := session.wait(ctx)
	return err
}

// resume starts a session from the latest saved state, unless one is already
// running.
func (p *sandboxImplPersistent) resume(ctx context.Context) error {
	if err := p.checkNotTerminated(); err != nil {
		return err
	}
	session := p.currentSession()
	sessionID, err := p.resumeSession(ctx)
	if err != nil {
		return err
	}
	if sessionID != session.id {
		p.mu.Lock()
		p.session = newSandboxImplV2(p.client, sessionID, sandboxImplParams{})
		p.mu.Unlock()
		session.close()
	}
	return nil
}

func (p *sandboxImplPersistent) close() {
	p.currentSession().close()
}

func (p *sandboxImplPersistent) wait(ctx context.Context) (int, error) {
	// A persistent Sandbox keeps running across sessions, until it is terminated.
	for !p.isTerminated() {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(persistentWaitPollInterval):
		}
	}
	return *getReturnCode(persistentTerminatedResult), nil
}

func (p *sandboxImplPersistent) poll(ctx context.Context) (*int, error) {
	if p.isTerminated() {
		return getReturnCode(persistentTerminatedResult), nil
	}
	return nil, nil
}

func (p *sandboxImplPersistent) waitForScheduling(ctx context.Context, timeout time.Duration) error {
	return p.currentSession().waitForScheduling(ctx, timeout)
}

// tunnels are addressed by the persistent Sandbox, not a session, so they are
// served without a running session and never go stale across sessions.
func (p *sandboxImplPersistent) tunnels(ctx context.Context, _ time.Duration) (map[int]*Tunnel, error) {
	resp, err := p.client.cpClient.PersistentSandboxTunnelsList(ctx, pb.PersistentSandboxTunnelsListRequest_builder{
		PersistentSandboxId: p.persistentSandboxID,
	}.Build())
	if err != nil {
		return nil, err
	}
	return tunnelsFromProtos(resp.GetTunnels()), nil
}

func (p *sandboxImplPersistent) getAppID(ctx context.Context) (string, error) {
	return run(ctx, p, func(s *sandboxImplV2) (string, error) { return s.getAppID(ctx) })
}

func (p *sandboxImplPersistent) resolveTaskIDForLogs(ctx context.Context) (string, error) {
	return p.currentSession().resolveTaskIDForLogs(ctx)
}

func (p *sandboxImplPersistent) getExitSnapshot(ctx context.Context, params *SandboxExperimentalGetExitSnapshotParams) (*Image, error) {
	return p.currentSession().getExitSnapshot(ctx, params)
}

func (p *sandboxImplPersistent) snapshotMemory(ctx context.Context) (*SandboxSnapshot, error) {
	return nil, InvalidError{Exception: "Memory snapshots aren't supported for persistent Sandboxes."}
}

func (p *sandboxImplPersistent) getCommandRouter(ctx context.Context) (string, *taskCommandRouterClient, error) {
	type router struct {
		taskID string
		client *taskCommandRouterClient
	}
	r, err := run(ctx, p, func(s *sandboxImplV2) (router, error) {
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

func (p *sandboxImplPersistent) snapshotFilesystem(ctx context.Context, timeout, ttl time.Duration, containerID string) (*Image, error) {
	return run(ctx, p, func(s *sandboxImplV2) (*Image, error) {
		return s.snapshotFilesystem(ctx, timeout, ttl, containerID)
	})
}

func (p *sandboxImplPersistent) snapshotDirectory(ctx context.Context, path string, timeout, ttl time.Duration, params *SandboxSnapshotDirectoryParams) (*Image, error) {
	return run(ctx, p, func(s *sandboxImplV2) (*Image, error) {
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
	return run(ctx, p, func(s *sandboxImplV2) (*ContainerProcess, error) {
		return s.exec(ctx, command, params, containerID)
	})
}

func (p *sandboxImplPersistent) validateSidecarCreate(params *SidecarCreateParams) error {
	return p.currentSession().validateSidecarCreate(params)
}

func (p *sandboxImplPersistent) createSidecar(ctx context.Context, in sidecarCreateInputs) (sidecarCreateResult, error) {
	return run(ctx, p, func(s *sandboxImplV2) (sidecarCreateResult, error) { return s.createSidecar(ctx, in) })
}
