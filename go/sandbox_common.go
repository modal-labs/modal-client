package modal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// How long Create waits for capacity before giving up.
const sandboxSchedulingTimeout = 21 * time.Minute

// How long operations that need a running task wait for one.
const getTaskIDTimeout = 5 * time.Minute

// sandboxImpl is the backend-specific implementation behind a Sandbox.
//
// A Sandbox is a thin facade that owns object identity and attachment state;
// everything that talks to a backend goes through one of these. Each backend
// (V1, V2) is a separate type, so the facade has no per-backend branching.
type sandboxImpl interface {
	sandboxID() string
	modalClient() *Client

	stdin() io.WriteCloser
	// outputStream returns one of the Sandbox's output streams. Reading it
	// after the Sandbox is detached is allowed where the backend supports it.
	outputStream(fd pb.FileDescriptor) io.ReadCloser

	getTags(ctx context.Context) (map[string]string, error)
	setTags(ctx context.Context, tags map[string]string) error
	setName(ctx context.Context, name string) error
	createConnectToken(ctx context.Context, port uint32, userMetadata string) (*SandboxCreateConnectCredentials, error)

	terminate(ctx context.Context) error
	// pause stops the current session of a persistent Sandbox, keeping its state.
	pause(ctx context.Context) error
	// resume resumes a paused persistent Sandbox from its saved state.
	resume(ctx context.Context) error
	// close releases local resources (e.g. connections); the Sandbox keeps running.
	close()
	wait(ctx context.Context) (int, error)
	poll(ctx context.Context) (*int, error)
	// waitForScheduling blocks until the Sandbox has a task. If the wait fails
	// for any reason other than the Sandbox having already finished, it
	// terminates the Sandbox so a queued one cannot start later with no caller.
	waitForScheduling(ctx context.Context, timeout time.Duration) error

	tunnels(ctx context.Context, timeout time.Duration) (map[int]*Tunnel, error)
	getAppID(ctx context.Context) (string, error)
	resolveTaskIDForLogs(ctx context.Context) (string, error)
	getExitSnapshot(ctx context.Context, params *SandboxExperimentalGetExitSnapshotParams) (*Image, error)
	snapshotMemory(ctx context.Context) (*SandboxSnapshot, error)

	getCommandRouter(ctx context.Context) (string, *taskCommandRouterClient, error)
	waitUntilReady(ctx context.Context, timeout time.Duration) error
	updateNetworkPolicy(ctx context.Context, params SandboxUpdateNetworkPolicyParams) error
	updateOutboundPolicy(ctx context.Context, policy *ExperimentalOutboundPolicy) error
	snapshotFilesystem(ctx context.Context, timeout, ttl time.Duration, containerID string) (*Image, error)
	snapshotDirectory(ctx context.Context, path string, timeout, ttl time.Duration, params *SandboxSnapshotDirectoryParams) (*Image, error)
	mountImage(ctx context.Context, path string, image *Image, params *SandboxMountImageParams) error
	unmountImage(ctx context.Context, path string) error
	// reloadVolumes reloads one container's Volumes; an empty containerID
	// targets the main container. A zero timeout selects the default.
	reloadVolumes(ctx context.Context, containerID string, timeout time.Duration) error
	exec(ctx context.Context, command []string, params *SandboxExecParams, containerID string) (*ContainerProcess, error)

	// validateSidecarCreate rejects sidecar options that createSidecar cannot
	// provide on this backend.
	validateSidecarCreate(params *SidecarCreateParams) error
	createSidecar(ctx context.Context, in sidecarCreateInputs) (sidecarCreateResult, error)
}

// sandboxImplParams seeds state a backend already knows at construction time.
type sandboxImplParams struct {
	taskID  string
	tunnels map[int]*Tunnel
	// Only good for the first command router connection; later ones fetch fresh access.
	initCommandRouterAccess *commandRouterAccess
}

// taskState is the state a backend keeps about the task running its Sandbox.
type taskState struct {
	taskID   string
	taskIDMu sync.Mutex
	tunnels  map[int]*Tunnel
	// Access returned at create or restore is only good for the first connection.
	initCommandRouterAccess *commandRouterAccess
	commandRouter           *lazyCommandRouterClient
}

// lazyCommandRouterClient connects to the task command router on first use and
// shares that connection until close.
type lazyCommandRouterClient struct {
	connect func(ctx context.Context, taskID string) (*taskCommandRouterClient, error)

	mu     sync.Mutex
	client *taskCommandRouterClient
	// open is cleared by close(); a connection finishing initialization
	// afterwards is released instead of kept.
	open atomic.Bool
}

func newLazyCommandRouterClient(connect func(ctx context.Context, taskID string) (*taskCommandRouterClient, error)) *lazyCommandRouterClient {
	l := &lazyCommandRouterClient{connect: connect}
	l.open.Store(true)
	return l
}

func (l *lazyCommandRouterClient) ensureOpen() error {
	if !l.open.Load() {
		return ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
	}
	return nil
}

// get returns the shared connection, connecting to task taskID first if needed.
func (l *lazyCommandRouterClient) get(ctx context.Context, taskID string) (*taskCommandRouterClient, error) {
	if err := l.ensureOpen(); err != nil {
		return nil, err
	}
	l.mu.Lock()
	defer func() {
		if !l.open.Load() && l.client != nil {
			l.client.closeWhenIdle()
			l.client = nil
		}
		l.mu.Unlock()
	}()

	if err := l.ensureOpen(); err != nil {
		return nil, err
	}

	if l.client == nil {
		client, err := l.connect(ctx, taskID)
		if err != nil {
			return nil, err
		}
		l.client = client
	}
	if err := l.ensureOpen(); err != nil {
		return nil, err
	}
	return l.client, nil
}

// close releases the connection, if any, and fails every later get.
func (l *lazyCommandRouterClient) close() {
	if !l.open.Swap(false) {
		return
	}
	// Initialization owns cleanup if it is still holding this lock.
	if !l.mu.TryLock() {
		return
	}
	defer l.mu.Unlock()
	if l.client != nil {
		l.client.closeWhenIdle()
		l.client = nil
	}
}

// connectCommandRouter opens a command router connection to task taskID of
// Sandbox sandboxID, using access if given and fetching access otherwise.
func connectCommandRouter(ctx context.Context, client *Client, taskID, sandboxID string, isV2 bool, access *commandRouterAccess) (*taskCommandRouterClient, error) {
	return initTaskCommandRouterClient(
		ctx,
		client.cpClient,
		taskID,
		sandboxID,
		isV2,
		access,
		client.logger,
		client.serverWarningLogger,
		client.profile,
	)
}

// waitForReturnCode long-polls rpcWait until the Sandbox finishes and returns
// its exit code.
func waitForReturnCode(ctx context.Context, impl sandboxImpl, rpcWait func(ctx context.Context, timeoutSecs float32) (*pb.SandboxWaitResponse, error)) (int, error) {
	for {
		if err := ctx.Err(); err != nil {
			return 0, err
		}

		resp, err := rpcWait(ctx, 10)
		if err != nil {
			return 0, err
		}
		if resp.GetResult() != nil {
			returnCode := getReturnCode(resp.GetResult())
			impl.modalClient().logger.DebugContext(ctx, "Sandbox wait completed",
				"sandbox_id", impl.sandboxID(),
				"status", resp.GetResult().GetStatus().String(),
				"return_code", returnCode)
			if returnCode != nil {
				return *returnCode, nil
			}
			return 0, nil
		}
	}
}

// waitForScheduling blocks until waitForTaskID returns a task. If the wait
// fails for any reason other than the Sandbox having already finished, it
// terminates the Sandbox so a queued one cannot start later with no caller.
func waitForScheduling(ctx context.Context, impl sandboxImpl, waitForTaskID func(ctx context.Context, timeout time.Duration) (string, error), timeout time.Duration) error {
	_, err := waitForTaskID(ctx, timeout)
	if err == nil {
		return nil
	}
	var conflictErr ConflictError
	if errors.As(err, &conflictErr) {
		return err
	}
	// The caller's context may already be done, so terminate on a detached one.
	if terminateErr := impl.terminate(context.WithoutCancel(ctx)); terminateErr != nil {
		impl.modalClient().logger.DebugContext(ctx, "Failed to terminate unscheduled Sandbox", "sandbox_id", impl.sandboxID(), "error", terminateErr)
	}
	var timeoutErr TimeoutError
	if errors.As(err, &timeoutErr) {
		return ResourceExhaustedError{Exception: "Insufficient capacity to create sandbox."}
	}
	return err
}

// tunnelsFromResponse converts a tunnels response into Tunnels by container port.
func tunnelsFromResponse(resp *pb.SandboxGetTunnelsResponse) (map[int]*Tunnel, error) {
	if resp.GetResult() != nil && resp.GetResult().GetStatus() == pb.GenericResult_GENERIC_STATUS_TIMEOUT {
		return nil, SandboxTimeoutError{Exception: "Sandbox operation timed out"}
	}
	return tunnelsFromProtos(resp.GetTunnels()), nil
}

// tunnelsFromProtos converts tunnel protos into Tunnels by container port.
func tunnelsFromProtos(protos []*pb.TunnelData) map[int]*Tunnel {
	tunnels := make(map[int]*Tunnel, len(protos))
	for _, tun := range protos {
		tunnels[int(tun.GetContainerPort())] = &Tunnel{
			Host:            tun.GetHost(),
			Port:            int(tun.GetPort()),
			UnencryptedHost: tun.GetUnencryptedHost(),
			UnencryptedPort: int(tun.GetUnencryptedPort()),
		}
	}
	return tunnels
}

// pollTaskID polls rpcGetTaskID until Sandbox sandboxID has been assigned a
// task. It returns a ConflictError if the Sandbox finished before that happened
// and a TimeoutError once timeout elapses. With retryTransient, lookups that
// fail with a transient error keep polling until the deadline instead of
// failing the wait.
func pollTaskID(ctx context.Context, sandboxID string, rpcGetTaskID func(ctx context.Context) (*pb.SandboxGetTaskIdResponse, error), timeout time.Duration, retryTransient bool) (string, error) {
	const pollInterval = 500 * time.Millisecond
	deadline := time.Now().Add(timeout)
	noTimeLeft := func() bool { return time.Until(deadline) <= pollInterval }
	timedOut := TimeoutError{Exception: fmt.Sprintf("timed out waiting for task ID for Sandbox %s", sandboxID)}
	for {
		resp, err := rpcGetTaskID(ctx)
		if err != nil {
			// A done caller context surfaces from the RPC as a gRPC status error, so check ctx directly.
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			if !isTransientTaskIDLookupError(err) || (!retryTransient && time.Now().Before(deadline)) {
				return "", err
			}
			if noTimeLeft() {
				// A server error is not evidence of missing capacity, so surface it as is.
				if status.Code(err) == codes.Internal {
					return "", err
				}
				return "", timedOut
			}
		} else if resp.GetTaskId() != "" {
			return resp.GetTaskId(), nil
		} else if result := resp.GetTaskResult(); result != nil {
			msg := result.GetException()
			if msg == "" {
				msg = fmt.Sprintf("Sandbox %s has already finished", sandboxID)
			}
			return "", ConflictError{Exception: msg}
		} else if noTimeLeft() {
			return "", timedOut
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}

// persistentOnlyMessage is the error for operations that only persistent
// Sandboxes support.
func persistentOnlyMessage(methodName string) string {
	return fmt.Sprintf("Sandbox.%s requires a persistent Sandbox, created with ExperimentalPersistent: true", methodName)
}

func isTransientTaskIDLookupError(err error) bool {
	switch status.Code(err) {
	case codes.Unavailable, codes.DeadlineExceeded, codes.Internal:
		return true
	}
	return false
}

// waitUntilReadyViaTcr blocks until the Sandbox reports ready over the command router.
func waitUntilReadyViaTcr(ctx context.Context, impl sandboxImpl, timeout time.Duration) error {
	taskID, commandRouterClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return err
	}
	_, err = commandRouterClient.SandboxWaitUntilReady(ctx, taskID, timeout)
	return err
}

// updateNetworkPolicyViaTcr replaces the network policy of a Sandbox over the command router.
func updateNetworkPolicyViaTcr(ctx context.Context, impl sandboxImpl, params SandboxUpdateNetworkPolicyParams) error {
	taskID, crClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return err
	}
	return crClient.SetNetworkAccess(ctx, buildUpdateNetworkPolicyProto(taskID, params))
}

// updateOutboundPolicyViaTcr replaces the outbound policy of a Sandbox over the command router.
func updateOutboundPolicyViaTcr(ctx context.Context, impl sandboxImpl, policy *ExperimentalOutboundPolicy) error {
	var taskID string
	var crClient *taskCommandRouterClient
	g, groupCtx := errgroup.WithContext(ctx)
	g.Go(func() error {
		return hydrateSecrets(groupCtx, impl.modalClient(), policy.secrets())
	})
	g.Go(func() error {
		var err error
		taskID, crClient, err = impl.getCommandRouter(groupCtx)
		return err
	})
	if err := g.Wait(); err != nil {
		return err
	}
	return crClient.SetOutboundPolicy(ctx, pb.TaskSetExperimentalOutboundPolicyRequest_builder{
		TaskId:                     taskID,
		ExperimentalOutboundPolicy: policy.toProto(),
	}.Build())
}

// snapshotFilesystemViaTcr snapshots one container's filesystem over the command router.
func snapshotFilesystemViaTcr(ctx context.Context, impl sandboxImpl, timeout, ttl time.Duration, containerID string) (*Image, error) {
	wireTTL, err := resolveTTL(ttl)
	if err != nil {
		return nil, err
	}

	taskID, crClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	request := pb.TaskSnapshotFilesystemRequest_builder{
		TaskId:      taskID,
		SnapshotId:  uuid.NewString(),
		TtlSeconds:  &wireTTL,
		ContainerId: containerID,
	}.Build()

	response, err := crClient.SnapshotFilesystem(ctx, request, timeout)
	if err != nil {
		return nil, err
	}

	if response.GetImageId() == "" {
		return nil, ExecutionError{Exception: "Sandbox snapshot filesystem response missing image ID"}
	}

	return &Image{ImageID: response.GetImageId(), client: impl.modalClient()}, nil
}

// snapshotDirectoryViaTcr snapshots one directory over the command router.
func snapshotDirectoryViaTcr(ctx context.Context, impl sandboxImpl, path string, timeout, ttl time.Duration, params *SandboxSnapshotDirectoryParams) (*Image, error) {
	wireTTL, err := resolveTTL(ttl)
	if err != nil {
		return nil, err
	}

	taskID, crClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	// SnapshotId guarantees idempotency under retries.
	request, err := buildTaskSnapshotDirectoryRequestProto(taskID, path, uuid.NewString(), "", wireTTL, params)
	if err != nil {
		return nil, err
	}

	response, err := crClient.SnapshotDirectory(ctx, request, timeout)
	if err != nil {
		return nil, err
	}

	if response.GetImageId() == "" {
		return nil, ExecutionError{Exception: "Sandbox snapshot directory response missing `imageId`"}
	}

	return &Image{ImageID: response.GetImageId(), client: impl.modalClient()}, nil
}

// mountImageViaTcr mounts an Image at path over the command router.
func mountImageViaTcr(ctx context.Context, impl sandboxImpl, path string, image *Image, params *SandboxMountImageParams) error {
	taskID, crClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return err
	}

	imageID, err := resolveMountImageID(image)
	if err != nil {
		return err
	}

	request, err := buildTaskMountDirectoryRequestProto(taskID, path, imageID, "", params)
	if err != nil {
		return err
	}

	return crClient.MountDirectory(ctx, request)
}

// unmountImageViaTcr unmounts the Image at path over the command router.
func unmountImageViaTcr(ctx context.Context, impl sandboxImpl, path string) error {
	taskID, crClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return err
	}

	request := pb.TaskUnmountDirectoryRequest_builder{
		TaskId:      taskID,
		Path:        []byte(path),
		ContainerId: "",
	}.Build()

	return crClient.UnmountDirectory(ctx, request)
}

// reloadVolumesViaTcr reloads one container's Volumes over the command router.
func reloadVolumesViaTcr(ctx context.Context, impl sandboxImpl, containerID string, timeout time.Duration) error {
	if timeout < 0 {
		return InvalidError{Exception: "Timeout must not be negative"}
	}
	if timeout == 0 {
		timeout = 55 * time.Second
	}
	taskID, crClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return err
	}
	return crClient.ReloadVolumes(ctx, pb.TaskReloadVolumesRequest_builder{
		TaskId:      taskID,
		ContainerId: containerID,
	}.Build(), timeout)
}

// execViaTcr starts a command in the Sandbox over the command router.
func execViaTcr(ctx context.Context, impl sandboxImpl, command []string, params *SandboxExecParams, containerID string) (*ContainerProcess, error) {
	if params == nil {
		params = &SandboxExecParams{}
	}

	if err := validateExecArgs(command); err != nil {
		return nil, err
	}

	// Locally-created Secrets (FromMap) are inlined into the request's
	// secret_sources, so only the remaining Secrets need hydrating. This avoids
	// a SecretGetOrCreate round-trip for env-dict Secrets.
	for k := range params.Env {
		if err := validateEnvVarName(k); err != nil {
			return nil, err
		}
	}
	if err := hydrateSecrets(ctx, impl.modalClient(), resolvableSecrets(params.Secrets)); err != nil {
		return nil, err
	}

	taskID, commandRouterClient, err := impl.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	execID := uuid.New().String()
	req, err := buildTaskExecStartRequestProto(taskID, execID, command, *params, containerID)
	if err != nil {
		return nil, err
	}

	_, err = commandRouterClient.ExecStart(ctx, req)
	if err != nil {
		return nil, err
	}

	impl.modalClient().logger.DebugContext(ctx, "Created ContainerProcess",
		"exec_id", execID,
		"sandbox_id", impl.sandboxID(),
		"container_id", containerID,
		"command", command)

	var deadline *time.Time
	if params.Timeout > 0 {
		d := time.Now().Add(params.Timeout)
		deadline = &d
	}

	return newContainerProcess(commandRouterClient, impl.modalClient().logger, taskID, execID, *params, deadline), nil
}

// createSidecarViaTcr creates a sidecar container over the Sandbox's task
// command router connection.
func createSidecarViaTcr(ctx context.Context, impl sandboxImpl, in sidecarCreateInputs) (sidecarCreateResult, error) {
	taskID, client, err := impl.getCommandRouter(ctx)
	if err != nil {
		return sidecarCreateResult{}, err
	}

	req := pb.TaskContainerCreateRequest_builder{
		TaskId:                  taskID,
		ContainerName:           in.name,
		ImageId:                 in.image.ImageID,
		Args:                    in.params.Command,
		Workdir:                 in.params.Workdir,
		SecretSources:           in.secretSources,
		VolumeMounts:            in.volumeMounts,
		NetworkAccess:           in.networkAccess,
		PtyInfo:                 in.ptyInfo,
		MemoryReserveConsumeMib: in.memoryReserveConsumeMib,
	}.Build()

	resp, err := client.ContainerCreate(ctx, req)
	if err != nil {
		return sidecarCreateResult{}, sidecarCreateError(err)
	}
	return sidecarCreateResult{containerID: resp.GetContainerId(), containerName: resp.GetContainerName()}, nil
}

// pollExitSnapshot polls rpcGetExitSnapshot until the exit snapshot of impl's
// Sandbox reaches a terminal state, within the budget of params.
func pollExitSnapshot(ctx context.Context, impl sandboxImpl, params *SandboxExperimentalGetExitSnapshotParams, rpcGetExitSnapshot func(ctx context.Context, timeoutSecs float32, opts ...grpc.CallOption) (*pb.SandboxGetExitSnapshotResponse, error)) (*Image, error) {
	sandboxID := impl.sandboxID()
	var timeout *time.Duration
	if params != nil {
		timeout = params.Timeout
	}
	if timeout != nil && *timeout < 0 {
		return nil, InvalidError{Exception: fmt.Sprintf("Timeout must be non-negative, got %v", *timeout)}
	}

	var deadline time.Time
	if timeout != nil {
		deadline = time.Now().Add(*timeout)
	}
	timeoutErr := TimeoutError{Exception: fmt.Sprintf("timed out waiting for exit snapshot for Sandbox %s", sandboxID)}
	pollRetries := 0
	consecutivePollFailures := 0
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		requestTimeout := exitSnapshotLongPollTimeout
		if timeout != nil {
			requestTimeout = min(exitSnapshotLongPollTimeout, max(0, time.Until(deadline)))
		}

		pollCtx, cancel := context.WithTimeout(ctx, requestTimeout+exitSnapshotPollDeadlineMargin)
		resp, err := rpcGetExitSnapshot(pollCtx, float32(requestTimeout.Seconds()), retryCallOption{retries: &pollRetries})
		cancel()
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			if policy := getServerRetryPolicy(err); policy != nil {
				if timeout != nil && !time.Now().Before(deadline) {
					return nil, timeoutErr
				}
				consecutivePollFailures = 0
				throttleDelay := max(time.Duration(float64(policy.GetRetryAfterSecs())*float64(time.Second)), defaultRetryBaseDelay)
				if timeout != nil {
					throttleDelay = min(throttleDelay, max(0, time.Until(deadline)))
				}
				if sleepErr := sleepCtx(ctx, throttleDelay); sleepErr != nil {
					return nil, sleepErr
				}
				continue
			}
			if st, ok := status.FromError(err); ok {
				switch st.Code() {
				case codes.InvalidArgument:
					return nil, InvalidError{Exception: st.Message()}
				case codes.NotFound:
					return nil, NotFoundError{Exception: st.Message()}
				}
			}
			if isRetryableGrpc(err) || errors.Is(err, context.DeadlineExceeded) {
				if timeout != nil && !time.Now().Before(deadline) {
					return nil, timeoutErr
				}
				consecutivePollFailures++
				if consecutivePollFailures >= exitSnapshotMaxConsecutivePollFailures {
					return nil, err
				}
				backoff := exitSnapshotPollFailureBackoff
				if timeout != nil {
					backoff = min(backoff, max(0, time.Until(deadline)))
				}
				if sleepErr := sleepCtx(ctx, backoff); sleepErr != nil {
					return nil, sleepErr
				}
				continue
			}
			return nil, err
		}
		consecutivePollFailures = 0

		switch {
		case resp.GetSuccess() != nil:
			if resp.GetSuccess().GetImageId() == "" {
				return nil, ExecutionError{Exception: "Exit snapshot result is missing image ID"}
			}
			return &Image{ImageID: resp.GetSuccess().GetImageId(), client: impl.modalClient()}, nil
		case resp.GetError() != nil:
			message := resp.GetError().GetMessage()
			errorCode := resp.GetError().GetErrorCode()
			if errorCode == pb.SandboxGetExitSnapshotResponse_ERROR_CODE_TIMEOUT ||
				errorCode == pb.SandboxGetExitSnapshotResponse_ERROR_CODE_FILESYSTEM_INCONSISTENT {
				if message == "" {
					message = "No exit snapshot image will be produced"
				}
				return nil, SnapshotCreationError{Exception: message}
			}
			if message == "" {
				message = "Exit snapshot state is malformed"
			}
			return nil, ExecutionError{Exception: message}
		case resp.GetPending() == nil:
			return nil, ExecutionError{Exception: "Exit snapshot response is missing an outcome"}
		}

		if timeout != nil && !time.Now().Before(deadline) {
			return nil, timeoutErr
		}
	}
}

// newSandboxImpl constructs the implementation backing the Sandbox with the
// given ID. Server responses can return a Sandbox of either backend regardless
// of which RPC was called, so the backend is always deduced from the ID.
func newSandboxImpl(client *Client, sandboxID string, params sandboxImplParams) sandboxImpl {
	if getSandboxVersion(sandboxID) == sandboxVersionV2 {
		return newSandboxImplV2(client, sandboxID, params)
	}
	return newSandboxImplV1(client, sandboxID, params)
}

func sandboxTagsToProto(tags map[string]string) []*pb.SandboxTag {
	tagsList := make([]*pb.SandboxTag, 0, len(tags))
	for k, v := range tags {
		tagsList = append(tagsList, pb.SandboxTag_builder{TagName: k, TagValue: v}.Build())
	}
	return tagsList
}

func sandboxTagsFromResponse(resp *pb.SandboxTagsGetResponse, err error) (map[string]string, error) {
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.InvalidArgument {
			return nil, InvalidError{Exception: status.Message()}
		}
		return nil, err
	}
	tags := make(map[string]string, len(resp.GetTags()))
	for _, tag := range resp.GetTags() {
		tags[tag.GetTagName()] = tag.GetTagValue()
	}
	return tags, nil
}

// task exposes the task state behind a Sandbox on a task-based backend. Used
// by tests to seed and inspect task state.
func (sb *Sandbox) task() *taskState {
	return sb.impl.(interface{ task() *taskState }).task()
}
