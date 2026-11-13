package modal

import (
	"context"
	"fmt"
	"io"
	"iter"
	"slices"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// sandboxImplV1 is the Sandbox backend whose control plane RPCs are the
// original (V1) Sandbox RPCs.
type sandboxImplV1 struct {
	client *Client
	id     string
	taskState
}

func newSandboxImplV1(client *Client, sandboxID string, params sandboxImplParams) *sandboxImplV1 {
	s := &sandboxImplV1{
		client: client,
		id:     sandboxID,
		taskState: taskState{
			taskID:                  params.taskID,
			tunnels:                 params.tunnels,
			initCommandRouterAccess: params.initCommandRouterAccess,
		},
	}
	s.commandRouter = newLazyCommandRouterClient(s.connectCommandRouter)
	return s
}

func (s *sandboxImplV1) sandboxID() string    { return s.id }
func (s *sandboxImplV1) modalClient() *Client { return s.client }
func (s *sandboxImplV1) task() *taskState     { return &s.taskState }

// createSandboxV1 creates a V1 Sandbox and waits for it to be scheduled.
func createSandboxV1(ctx context.Context, client *Client, app *App, image *Image, params *SandboxCreateParams) (sandboxImpl, error) {
	image, err := image.Build(ctx, app, nil)
	if err != nil {
		return nil, err
	}

	mergedSecrets, err := mergeEnvIntoSecrets(ctx, client, &params.Env, &params.Secrets)
	if err != nil {
		return nil, err
	}

	// The SandboxCreate request only carries secret IDs, so any locally-created
	// Secrets (and env vars) must be hydrated into server-side Secrets first.
	if err := hydrateSandboxSecrets(ctx, client, append(slices.Clone(mergedSecrets), params.ExperimentalOutboundPolicy.secrets()...), params.CloudBucketMounts); err != nil {
		return nil, err
	}

	mergedParams := *params
	mergedParams.Secrets = mergedSecrets
	mergedParams.Env = nil // nil'ing Env just to clarify it's not needed anymore

	req, err := buildSandboxCreateRequestProto(app.AppID, image.ImageID, mergedParams)
	if err != nil {
		return nil, err
	}

	createResp, err := client.cpClient.SandboxCreate(ctx, req)
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.AlreadyExists {
			return nil, AlreadyExistsError{Exception: status.Message()}
		}
		return nil, err
	}

	client.logger.DebugContext(ctx, "Created Sandbox", "sandbox_id", createResp.GetSandboxId())
	impl := newSandboxImpl(client, createResp.GetSandboxId(), sandboxImplParams{})
	if err := impl.waitForScheduling(ctx, sandboxSchedulingTimeout); err != nil {
		return nil, err
	}
	return impl, nil
}

// sandboxFromNameV1 looks up a running Sandbox by name through the V1 RPC.
func sandboxFromNameV1(ctx context.Context, client *Client, appName, name string, params *SandboxFromNameParams) (sandboxImpl, error) {
	resp, err := client.cpClient.SandboxGetFromName(ctx, pb.SandboxGetFromNameRequest_builder{
		SandboxName:     name,
		AppName:         appName,
		EnvironmentName: firstNonEmpty(params.Environment, client.profile.Environment),
	}.Build())
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
			return nil, NotFoundError{Exception: fmt.Sprintf("Sandbox with name '%s' not found in App '%s'", name, appName)}
		}
		return nil, err
	}
	return newSandboxImpl(client, resp.GetSandboxId(), sandboxImplParams{}), nil
}

// listSandboxesV1 lists running Sandboxes through the V1 list RPC.
func listSandboxesV1(ctx context.Context, client *Client, params *SandboxListParams) iter.Seq2[sandboxImpl, error] {
	tagsList := make([]*pb.SandboxTag, 0, len(params.Tags))
	for k, v := range params.Tags {
		tagsList = append(tagsList, pb.SandboxTag_builder{TagName: k, TagValue: v}.Build())
	}

	return func(yield func(sandboxImpl, error) bool) {
		var before float64
		for {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			resp, err := client.cpClient.SandboxList(ctx, pb.SandboxListRequest_builder{
				AppId:           params.AppID,
				BeforeTimestamp: before,
				EnvironmentName: firstNonEmpty(params.Environment, client.profile.Environment),
				IncludeFinished: false,
				Tags:            tagsList,
			}.Build())
			if err != nil {
				yield(nil, err)
				return
			}
			sandboxes := resp.GetSandboxes()
			if len(sandboxes) == 0 {
				return
			}
			for _, info := range sandboxes {
				if !yield(newSandboxImpl(client, info.GetId(), sandboxImplParams{}), nil) {
					return
				}
			}
			before = sandboxes[len(sandboxes)-1].GetCreatedAt()
		}
	}
}

// sandboxFromSnapshotV1 restores a V1 memory snapshot into a new Sandbox and
// waits for it to be ready.
func sandboxFromSnapshotV1(ctx context.Context, client *Client, snapshot *SandboxSnapshot, overrideType pb.SandboxRestoreRequest_SandboxNameOverrideType, nameOverride string) (sandboxImpl, error) {
	resp, err := client.cpClient.SandboxRestore(ctx, pb.SandboxRestoreRequest_builder{
		SnapshotId:              snapshot.SnapshotID,
		SandboxNameOverride:     nameOverride,
		SandboxNameOverrideType: overrideType,
	}.Build())
	if err != nil {
		return nil, err
	}

	impl := newSandboxImpl(client, resp.GetSandboxId(), sandboxImplParams{})

	timeoutSecs := float32(55)
	taskResp, err := client.cpClient.SandboxGetTaskId(ctx, pb.SandboxGetTaskIdRequest_builder{
		SandboxId:      resp.GetSandboxId(),
		WaitUntilReady: true,
		Timeout:        &timeoutSecs,
	}.Build())
	if err != nil {
		return nil, err
	}
	if result := taskResp.GetTaskResult(); result != nil &&
		result.GetStatus() != pb.GenericResult_GENERIC_STATUS_UNSPECIFIED &&
		result.GetStatus() != pb.GenericResult_GENERIC_STATUS_SUCCESS {
		return nil, ExecutionError{Exception: result.GetException()}
	}
	return impl, nil
}

func (s *sandboxImplV1) terminate(ctx context.Context) error {
	_, err := s.client.cpClient.SandboxTerminate(ctx, pb.SandboxTerminateRequest_builder{SandboxId: s.id}.Build())
	return err
}

func (s *sandboxImplV1) pause(ctx context.Context) error {
	return InvalidError{Exception: persistentOnlyMessage("ExperimentalPause")}
}

func (s *sandboxImplV1) resume(ctx context.Context) error {
	return InvalidError{Exception: persistentOnlyMessage("ExperimentalResume")}
}

func (s *sandboxImplV1) close() {
	s.commandRouter.close()
}

func (s *sandboxImplV1) rpcWait(ctx context.Context, timeoutSecs float32) (*pb.SandboxWaitResponse, error) {
	return s.client.cpClient.SandboxWait(ctx, pb.SandboxWaitRequest_builder{SandboxId: s.id, Timeout: timeoutSecs}.Build())
}

func (s *sandboxImplV1) wait(ctx context.Context) (int, error) {
	return waitForReturnCode(ctx, s, s.rpcWait)
}

func (s *sandboxImplV1) poll(ctx context.Context) (*int, error) {
	resp, err := s.rpcWait(ctx, 0)
	if err != nil {
		return nil, err
	}
	return getReturnCode(resp.GetResult()), nil
}

func (s *sandboxImplV1) waitForScheduling(ctx context.Context, timeout time.Duration) error {
	return waitForScheduling(ctx, s, func(ctx context.Context, timeout time.Duration) (string, error) {
		return s.waitForTaskID(ctx, timeout, true)
	}, timeout)
}

func (s *sandboxImplV1) tunnels(ctx context.Context, timeout time.Duration) (map[int]*Tunnel, error) {
	if s.taskState.tunnels != nil {
		return s.taskState.tunnels, nil
	}
	resp, err := s.client.cpClient.SandboxGetTunnels(ctx, pb.SandboxGetTunnelsRequest_builder{
		SandboxId: s.id,
		Timeout:   float32(timeout.Seconds()),
	}.Build())
	if err != nil {
		return nil, err
	}
	tunnels, err := tunnelsFromResponse(resp)
	if err != nil {
		return nil, err
	}
	s.taskState.tunnels = tunnels
	return tunnels, nil
}

func (s *sandboxImplV1) getAppID(ctx context.Context) (string, error) {
	resp, err := s.rpcWait(ctx, 0)
	return resp.GetMetadata().GetAppId(), err
}

func (s *sandboxImplV1) rpcGetTaskID(ctx context.Context) (*pb.SandboxGetTaskIdResponse, error) {
	return s.client.cpClient.SandboxGetTaskId(ctx, pb.SandboxGetTaskIdRequest_builder{SandboxId: s.id}.Build())
}

func (s *sandboxImplV1) resolveTaskIDForLogs(ctx context.Context) (string, error) {
	resp, err := s.rpcGetTaskID(ctx)
	return resp.GetTaskId(), err
}

func (s *sandboxImplV1) ensureTaskID(ctx context.Context) (string, error) {
	return s.waitForTaskID(ctx, getTaskIDTimeout, false)
}

// waitForTaskID returns the Sandbox's task ID, polling for it until timeout
// if it is not known yet.
func (s *sandboxImplV1) waitForTaskID(ctx context.Context, timeout time.Duration, retryTransient bool) (string, error) {
	s.taskIDMu.Lock()
	taskID := s.taskID
	s.taskIDMu.Unlock()
	if taskID != "" {
		return taskID, nil
	}
	taskID, err := pollTaskID(ctx, s.id, s.rpcGetTaskID, timeout, retryTransient)
	if err != nil {
		return "", err
	}
	s.taskIDMu.Lock()
	s.taskID = taskID
	s.taskIDMu.Unlock()
	return taskID, nil
}

func (s *sandboxImplV1) getExitSnapshot(ctx context.Context, params *SandboxExperimentalGetExitSnapshotParams) (*Image, error) {
	return pollExitSnapshot(ctx, s, params, func(ctx context.Context, timeoutSecs float32, opts ...grpc.CallOption) (*pb.SandboxGetExitSnapshotResponse, error) {
		return s.client.cpClient.SandboxGetExitSnapshot(ctx, pb.SandboxGetExitSnapshotRequest_builder{
			SandboxId: s.id,
			Timeout:   timeoutSecs,
		}.Build(), opts...)
	})
}

func (s *sandboxImplV1) getCommandRouter(ctx context.Context) (string, *taskCommandRouterClient, error) {
	if err := s.commandRouter.ensureOpen(); err != nil {
		return "", nil, err
	}
	taskID, err := s.ensureTaskID(ctx)
	if err != nil {
		return "", nil, err
	}
	client, err := s.commandRouter.get(ctx, taskID)
	if err != nil {
		return "", nil, err
	}
	return taskID, client, nil
}

func (s *sandboxImplV1) connectCommandRouter(ctx context.Context, taskID string) (*taskCommandRouterClient, error) {
	// Consume the seeded access, so if connecting with it fails the retry
	// falls back to the authoritative RPC instead of re-trying the same
	// credentials forever. Safe under the lazy client's lock, held by get.
	access := s.initCommandRouterAccess
	s.initCommandRouterAccess = nil
	return connectCommandRouter(ctx, s.client, taskID, s.id, false, access)
}

func (s *sandboxImplV1) waitUntilReady(ctx context.Context, timeout time.Duration) error {
	return waitUntilReadyViaTcr(ctx, s, timeout)
}

func (s *sandboxImplV1) updateNetworkPolicy(ctx context.Context, params SandboxUpdateNetworkPolicyParams) error {
	return updateNetworkPolicyViaTcr(ctx, s, params)
}

func (s *sandboxImplV1) updateOutboundPolicy(ctx context.Context, policy *ExperimentalOutboundPolicy) error {
	return updateOutboundPolicyViaTcr(ctx, s, policy)
}

func (s *sandboxImplV1) snapshotFilesystem(ctx context.Context, timeout, ttl time.Duration, containerID string) (*Image, error) {
	return snapshotFilesystemViaTcr(ctx, s, timeout, ttl, containerID)
}

func (s *sandboxImplV1) snapshotDirectory(ctx context.Context, path string, timeout, ttl time.Duration, params *SandboxSnapshotDirectoryParams) (*Image, error) {
	return snapshotDirectoryViaTcr(ctx, s, path, timeout, ttl, params)
}

func (s *sandboxImplV1) mountImage(ctx context.Context, path string, image *Image, params *SandboxMountImageParams) error {
	return mountImageViaTcr(ctx, s, path, image, params)
}

func (s *sandboxImplV1) unmountImage(ctx context.Context, path string) error {
	return unmountImageViaTcr(ctx, s, path)
}

func (s *sandboxImplV1) reloadVolumes(ctx context.Context, containerID string, timeout time.Duration) error {
	return reloadVolumesViaTcr(ctx, s, containerID, timeout)
}

func (s *sandboxImplV1) exec(ctx context.Context, command []string, params *SandboxExecParams, containerID string) (*ContainerProcess, error) {
	return execViaTcr(ctx, s, command, params, containerID)
}

func (s *sandboxImplV1) stdin() io.WriteCloser {
	return inputStreamSb(s.client.cpClient, s.id)
}

func (s *sandboxImplV1) outputStream(fd pb.FileDescriptor) io.ReadCloser {
	return outputStreamSb(s.client.cpClient, s.id, fd, s.client.profile.SandboxChannelIdleTimeout)
}

func (s *sandboxImplV1) setTags(ctx context.Context, tags map[string]string) error {
	_, err := s.client.cpClient.SandboxTagsSet(ctx, pb.SandboxTagsSetRequest_builder{
		EnvironmentName: s.client.profile.Environment,
		SandboxId:       s.id,
		Tags:            sandboxTagsToProto(tags),
	}.Build())
	return err
}

func (s *sandboxImplV1) getTags(ctx context.Context) (map[string]string, error) {
	resp, err := s.client.cpClient.SandboxTagsGet(ctx, pb.SandboxTagsGetRequest_builder{SandboxId: s.id}.Build())
	return sandboxTagsFromResponse(resp, err)
}

func (s *sandboxImplV1) setName(ctx context.Context, name string) error {
	return InvalidError{Exception: "Sandbox.ExperimentalSetName is only supported for V2 sandboxes"}
}

func (s *sandboxImplV1) createConnectToken(ctx context.Context, port uint32, userMetadata string) (*SandboxCreateConnectCredentials, error) {
	resp, err := s.client.cpClient.SandboxCreateConnectToken(ctx, pb.SandboxCreateConnectTokenRequest_builder{
		SandboxId:    s.id,
		UserMetadata: userMetadata,
		Port:         &port,
	}.Build())
	if err != nil {
		return nil, err
	}
	return &SandboxCreateConnectCredentials{URL: resp.GetUrl(), Token: resp.GetToken()}, nil
}

func (s *sandboxImplV1) snapshotMemory(ctx context.Context) (*SandboxSnapshot, error) {
	if _, err := s.ensureTaskID(ctx); err != nil {
		return nil, err
	}
	snapResp, err := s.client.cpClient.SandboxSnapshot(ctx, pb.SandboxSnapshotRequest_builder{
		SandboxId: s.id,
	}.Build())
	if err != nil {
		return nil, err
	}
	snapshotID := snapResp.GetSnapshotId()

	// wait for the snapshot to succeed. this is implemented as a second idempotent rpc
	// because the snapshot itself may take a while to complete.
	waitResp, err := s.client.cpClient.SandboxSnapshotWait(ctx, pb.SandboxSnapshotWaitRequest_builder{
		SnapshotId: snapshotID,
		Timeout:    55,
	}.Build())
	if err != nil {
		return nil, err
	}
	if waitResp.GetResult().GetStatus() != pb.GenericResult_GENERIC_STATUS_SUCCESS {
		return nil, ExecutionError{Exception: waitResp.GetResult().GetException()}
	}

	isV2 := false
	return &SandboxSnapshot{SnapshotID: snapshotID, isV2: &isV2, client: s.client}, nil
}

func (s *sandboxImplV1) validateSidecarCreate(params *SidecarCreateParams) error {
	if len(params.CloudBucketMounts) > 0 {
		return InvalidError{Exception: "CloudBucketMounts are not supported in sidecars of V1 Sandboxes. A Sandbox is V1 when it has a GPU or MODAL_SANDBOX_V2=0 is set; contact Modal support for more information"}
	}
	if params.IncludeOidcIdentityToken {
		return InvalidError{Exception: "IncludeOidcIdentityToken is not supported for GPU Sandboxes"}
	}
	if params.Proxy != nil {
		return InvalidError{Exception: "Proxy is not supported for GPU Sandboxes"}
	}
	return nil
}

func (s *sandboxImplV1) createSidecar(ctx context.Context, in sidecarCreateInputs) (sidecarCreateResult, error) {
	return createSidecarViaTcr(ctx, s, in)
}
