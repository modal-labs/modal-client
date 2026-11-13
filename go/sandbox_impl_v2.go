package modal

import (
	"context"
	"fmt"
	"io"
	"iter"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/google/uuid"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// controlPlaneSidecarCreateEnvVar opts a client out of sending sidecar create
// requests to the Modal server: set it to "0" or "false" to create sidecars
// over the Sandbox connection instead.
const controlPlaneSidecarCreateEnvVar = "MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE"

// controlPlaneSidecarCreateEnabled reports whether sidecar create sends its
// request to the Modal server, which V2 Sandboxes do unless opted out.
func controlPlaneSidecarCreateEnabled() bool {
	value, isSet := os.LookupEnv(controlPlaneSidecarCreateEnvVar)
	if !isSet {
		return true
	}
	return parseBooleanFlag(value)
}

// sandboxImplV2 is the Sandbox backend whose control plane RPCs are the V2
// Sandbox RPCs and whose stdio goes through the task command router.
type sandboxImplV2 struct {
	client *Client
	id     string
	taskState
}

func newSandboxImplV2(client *Client, sandboxID string, params sandboxImplParams) *sandboxImplV2 {
	s := &sandboxImplV2{
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

func (s *sandboxImplV2) sandboxID() string    { return s.id }
func (s *sandboxImplV2) modalClient() *Client { return s.client }
func (s *sandboxImplV2) task() *taskState     { return &s.taskState }

// createSandboxV2 creates a V2 Sandbox.
func createSandboxV2(ctx context.Context, client *Client, app *App, image *Image, params *SandboxCreateParams) (sandboxImpl, error) {
	image, err := image.Build(ctx, app, nil)
	if err != nil {
		return nil, err
	}

	// V2 takes Secrets and env vars as ordered secret_sources, so unlike the V1
	// backend we don't fold env vars into a server-side Secret. Locally-created
	// Secrets (FromMap) are inlined, avoiding a SecretGetOrCreate round-trip;
	// only the remaining resolvable Secrets (e.g. from FromName) need hydrating
	// to secret IDs, which are also recorded on the definition.
	for k := range params.Env {
		if err := validateEnvVarName(k); err != nil {
			return nil, err
		}
	}
	resolvable := resolvableSecrets(params.Secrets)
	if err := hydrateSandboxSecrets(ctx, client, append(slices.Clone(resolvable), params.ExperimentalOutboundPolicy.secrets()...), params.CloudBucketMounts); err != nil {
		return nil, err
	}
	sources, err := secretSources(params.Secrets, params.Env)
	if err != nil {
		return nil, err
	}

	definitionParams := *params
	definitionParams.Secrets = resolvable

	req, err := buildSandboxCreateV2RequestProto(app.AppID, image.ImageID, definitionParams, sources)
	if err != nil {
		return nil, err
	}

	createResp, err := client.cpClient.SandboxCreateV2(ctx, req)
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.AlreadyExists {
			return nil, AlreadyExistsError{Exception: status.Message()}
		}
		return nil, err
	}

	opts := sandboxImplParams{taskID: createResp.GetTaskId()}
	if access := createResp.GetCommandRouterAccess(); access != nil {
		opts.initCommandRouterAccess = &commandRouterAccess{jwt: access.GetJwt(), url: access.GetUrl()}
	}
	// Unencrypted tunnel endpoints are not known at create time. Only cache a
	// complete set so Tunnels() fetches the rest otherwise.
	numTunnels := len(createResp.GetTunnels())
	createRespHasAllTunnels := numTunnels > 0 &&
		numTunnels == len(req.GetDefinition().GetOpenPorts().GetPorts())
	if createRespHasAllTunnels {
		opts.tunnels = make(map[int]*Tunnel, numTunnels)
		for _, t := range createResp.GetTunnels() {
			opts.tunnels[int(t.GetContainerPort())] = &Tunnel{
				Host:            t.GetHost(),
				Port:            int(t.GetPort()),
				UnencryptedHost: t.GetUnencryptedHost(),
				UnencryptedPort: int(t.GetUnencryptedPort()),
			}
		}
	}

	client.logger.DebugContext(ctx, "Created experimental V2 Sandbox", "sandbox_id", createResp.GetSandboxId())
	return newSandboxImpl(client, createResp.GetSandboxId(), opts), nil
}

// sandboxFromNameV2 looks up a Sandbox by name through the V2 RPC.
func sandboxFromNameV2(ctx context.Context, client *Client, appName, name string, params *SandboxExperimentalFromNameParams) (sandboxImpl, error) {
	resp, err := client.cpClient.SandboxGetFromNameV2(ctx, pb.SandboxGetFromNameRequest_builder{
		SandboxName:       name,
		AppName:           appName,
		EnvironmentName:   firstNonEmpty(params.Environment, client.profile.Environment),
		IncludeTerminated: params.IncludeTerminated,
	}.Build())
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
			return nil, NotFoundError{Exception: fmt.Sprintf("Sandbox with name '%s' not found in App '%s'", name, appName)}
		}
		return nil, err
	}
	return newSandboxImpl(client, resp.GetSandboxId(), sandboxImplParams{}), nil
}

// listSandboxesV2 lists running Sandboxes through the V2 list RPC.
func listSandboxesV2(ctx context.Context, client *Client, params *SandboxExperimentalListParams) iter.Seq2[sandboxImpl, error] {
	var environmentName string
	if params.AppID == "" {
		client.logger.WarnContext(ctx,
			"Sandboxes.ExperimentalList without an AppID lists across the whole environment and is deprecated; pass AppID to scope the query")
		environmentName = firstNonEmpty(params.Environment, client.profile.Environment)
	}

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

			resp, err := client.cpClient.SandboxListV2(ctx, pb.SandboxListRequest_builder{
				AppId:           params.AppID,
				BeforeTimestamp: before,
				EnvironmentName: environmentName,
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

// sandboxFromSnapshotV2 restores a V2 memory snapshot into a new Sandbox.
func sandboxFromSnapshotV2(ctx context.Context, client *Client, snapshot *SandboxSnapshot, overrideType pb.SandboxRestoreRequest_SandboxNameOverrideType, nameOverride string) (sandboxImpl, error) {
	resp, err := client.cpClient.SandboxRestoreV2(ctx, pb.SandboxRestoreV2Request_builder{
		SnapshotId:              snapshot.SnapshotID,
		SandboxNameOverride:     nameOverride,
		SandboxNameOverrideType: overrideType,
	}.Build())
	if err != nil {
		return nil, err
	}
	opts := sandboxImplParams{taskID: resp.GetTaskId()}
	if access := resp.GetCommandRouterAccess(); access != nil {
		opts.initCommandRouterAccess = &commandRouterAccess{jwt: access.GetJwt(), url: access.GetUrl()}
	}
	return newSandboxImpl(client, resp.GetSandboxId(), opts), nil
}

func (s *sandboxImplV2) terminate(ctx context.Context) error {
	_, err := s.client.cpClient.SandboxTerminateV2(ctx, pb.SandboxTerminateRequest_builder{SandboxId: s.id}.Build())
	return err
}

func (s *sandboxImplV2) pause(ctx context.Context) error {
	return InvalidError{Exception: persistentOnlyMessage("ExperimentalPause")}
}

func (s *sandboxImplV2) resume(ctx context.Context) error {
	return InvalidError{Exception: persistentOnlyMessage("ExperimentalResume")}
}

func (s *sandboxImplV2) close() {
	s.commandRouter.close()
}

func (s *sandboxImplV2) rpcWait(ctx context.Context, timeoutSecs float32) (*pb.SandboxWaitResponse, error) {
	return s.client.cpClient.SandboxWaitV2(ctx, pb.SandboxWaitRequest_builder{SandboxId: s.id, Timeout: timeoutSecs}.Build())
}

func (s *sandboxImplV2) wait(ctx context.Context) (int, error) {
	return waitForReturnCode(ctx, s, s.rpcWait)
}

func (s *sandboxImplV2) poll(ctx context.Context) (*int, error) {
	resp, err := s.rpcWait(ctx, 0)
	if err != nil {
		return nil, err
	}
	return getReturnCode(resp.GetResult()), nil
}

func (s *sandboxImplV2) waitForScheduling(ctx context.Context, timeout time.Duration) error {
	return waitForScheduling(ctx, s, func(ctx context.Context, timeout time.Duration) (string, error) {
		return s.waitForTaskID(ctx, timeout, true)
	}, timeout)
}

func (s *sandboxImplV2) tunnels(ctx context.Context, timeout time.Duration) (map[int]*Tunnel, error) {
	if s.taskState.tunnels != nil {
		return s.taskState.tunnels, nil
	}
	resp, err := s.client.cpClient.SandboxGetTunnelsV2(ctx, pb.SandboxGetTunnelsRequest_builder{
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

func (s *sandboxImplV2) getAppID(ctx context.Context) (string, error) {
	resp, err := s.rpcWait(ctx, 0)
	return resp.GetMetadata().GetAppId(), err
}

func (s *sandboxImplV2) rpcGetTaskID(ctx context.Context) (*pb.SandboxGetTaskIdResponse, error) {
	return s.client.cpClient.SandboxGetTaskIdV2(ctx, pb.SandboxGetTaskIdRequest_builder{SandboxId: s.id}.Build())
}

func (s *sandboxImplV2) resolveTaskIDForLogs(ctx context.Context) (string, error) {
	resp, err := s.rpcGetTaskID(ctx)
	return resp.GetTaskId(), err
}

func (s *sandboxImplV2) ensureTaskID(ctx context.Context) (string, error) {
	return s.waitForTaskID(ctx, getTaskIDTimeout, false)
}

// waitForTaskID returns the Sandbox's task ID, polling for it until timeout
// if it is not known yet.
func (s *sandboxImplV2) waitForTaskID(ctx context.Context, timeout time.Duration, retryTransient bool) (string, error) {
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

func (s *sandboxImplV2) getExitSnapshot(ctx context.Context, params *SandboxExperimentalGetExitSnapshotParams) (*Image, error) {
	return pollExitSnapshot(ctx, s, params, func(ctx context.Context, timeoutSecs float32, opts ...grpc.CallOption) (*pb.SandboxGetExitSnapshotResponse, error) {
		return s.client.cpClient.SandboxGetExitSnapshotV2(ctx, pb.SandboxGetExitSnapshotRequest_builder{
			SandboxId: s.id,
			Timeout:   timeoutSecs,
		}.Build(), opts...)
	})
}

func (s *sandboxImplV2) getCommandRouter(ctx context.Context) (string, *taskCommandRouterClient, error) {
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

func (s *sandboxImplV2) connectCommandRouter(ctx context.Context, taskID string) (*taskCommandRouterClient, error) {
	// Consume the seeded access, so if connecting with it fails the retry
	// falls back to the authoritative RPC instead of re-trying the same
	// credentials forever. Safe under the lazy client's lock, held by get.
	access := s.initCommandRouterAccess
	s.initCommandRouterAccess = nil
	return connectCommandRouter(ctx, s.client, taskID, s.id, true, access)
}

func (s *sandboxImplV2) waitUntilReady(ctx context.Context, timeout time.Duration) error {
	return waitUntilReadyViaTcr(ctx, s, timeout)
}

func (s *sandboxImplV2) updateNetworkPolicy(ctx context.Context, params SandboxUpdateNetworkPolicyParams) error {
	return updateNetworkPolicyViaTcr(ctx, s, params)
}

func (s *sandboxImplV2) updateOutboundPolicy(ctx context.Context, policy *ExperimentalOutboundPolicy) error {
	return updateOutboundPolicyViaTcr(ctx, s, policy)
}

func (s *sandboxImplV2) snapshotFilesystem(ctx context.Context, timeout, ttl time.Duration, containerID string) (*Image, error) {
	return snapshotFilesystemViaTcr(ctx, s, timeout, ttl, containerID)
}

func (s *sandboxImplV2) snapshotDirectory(ctx context.Context, path string, timeout, ttl time.Duration, params *SandboxSnapshotDirectoryParams) (*Image, error) {
	return snapshotDirectoryViaTcr(ctx, s, path, timeout, ttl, params)
}

func (s *sandboxImplV2) mountImage(ctx context.Context, path string, image *Image, params *SandboxMountImageParams) error {
	return mountImageViaTcr(ctx, s, path, image, params)
}

func (s *sandboxImplV2) unmountImage(ctx context.Context, path string) error {
	return unmountImageViaTcr(ctx, s, path)
}

func (s *sandboxImplV2) reloadVolumes(ctx context.Context, containerID string, timeout time.Duration) error {
	return reloadVolumesViaTcr(ctx, s, containerID, timeout)
}

func (s *sandboxImplV2) exec(ctx context.Context, command []string, params *SandboxExecParams, containerID string) (*ContainerProcess, error) {
	return execViaTcr(ctx, s, command, params, containerID)
}

func (s *sandboxImplV2) stdin() io.WriteCloser {
	return &sbStdinV2{getCommandRouter: s.getCommandRouter}
}

// outputStream connects through the command router on first read, so
// creating a handle doesn't open a connection. Reads fail once the Sandbox is
// detached, since the connection is released then.
func (s *sandboxImplV2) outputStream(fd pb.FileDescriptor) io.ReadCloser {
	ctx, cancel := context.WithCancel(context.Background())
	return &lazyStreamReader{
		readCheck: s.commandRouter.ensureOpen,
		cancel:    cancel,
		initFunc: func() io.ReadCloser {
			return outputStreamSbV2(ctx, s.getCommandRouter, fd)
		},
	}
}

func (s *sandboxImplV2) setTags(ctx context.Context, tags map[string]string) error {
	_, err := s.client.cpClient.SandboxTagsSetV2(ctx, pb.SandboxTagsSetRequest_builder{
		SandboxId: s.id,
		Tags:      sandboxTagsToProto(tags),
	}.Build())
	return err
}

func (s *sandboxImplV2) getTags(ctx context.Context) (map[string]string, error) {
	resp, err := s.client.cpClient.SandboxTagsGetV2(ctx, pb.SandboxTagsGetRequest_builder{SandboxId: s.id}.Build())
	return sandboxTagsFromResponse(resp, err)
}

func (s *sandboxImplV2) setName(ctx context.Context, name string) error {
	if err := checkObjectName(name, "Sandbox"); err != nil {
		return err
	}

	_, err := s.client.cpClient.SandboxSetName(ctx, pb.SandboxSetNameRequest_builder{
		SandboxId: s.id,
		Name:      name,
	}.Build())
	if err != nil {
		if st, ok := status.FromError(err); ok {
			switch st.Code() {
			case codes.AlreadyExists:
				return AlreadyExistsError{Exception: st.Message()}
			case codes.InvalidArgument:
				return InvalidError{Exception: st.Message()}
			case codes.FailedPrecondition:
				return ConflictError{Exception: st.Message()}
			}
		}
		return err
	}
	return nil
}

func (s *sandboxImplV2) createConnectToken(ctx context.Context, port uint32, userMetadata string) (*SandboxCreateConnectCredentials, error) {
	resp, err := s.client.cpClient.SandboxCreateConnectTokenV2(ctx, pb.SandboxCreateConnectTokenRequest_builder{
		SandboxId:    s.id,
		UserMetadata: userMetadata,
		Port:         &port,
	}.Build())
	if err != nil {
		return nil, err
	}
	return &SandboxCreateConnectCredentials{URL: resp.GetUrl(), Token: resp.GetToken()}, nil
}

func (s *sandboxImplV2) snapshotMemory(ctx context.Context) (*SandboxSnapshot, error) {
	taskID, crClient, err := s.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}
	resp, err := crClient.SnapshotMemory(ctx, pb.TaskSnapshotMemoryRequest_builder{
		TaskId:         taskID,
		IdempotencyKey: uuid.NewString(),
	}.Build(), 300*time.Second)
	if err != nil {
		return nil, err
	}
	isV2 := true
	return &SandboxSnapshot{SnapshotID: resp.GetSnapshotId(), isV2: &isV2, client: s.client}, nil
}

func (s *sandboxImplV2) validateSidecarCreate(params *SidecarCreateParams) error {
	if controlPlaneSidecarCreateEnabled() {
		return nil
	}
	if len(params.CloudBucketMounts) > 0 {
		return InvalidError{Exception: fmt.Sprintf(
			"CloudBucketMounts are not supported in sidecars when %s=0 is set; unset it to use them",
			controlPlaneSidecarCreateEnvVar,
		)}
	}
	if params.IncludeOidcIdentityToken {
		return InvalidError{Exception: fmt.Sprintf(
			"IncludeOidcIdentityToken is not supported when %s=0 is set; unset it to use it",
			controlPlaneSidecarCreateEnvVar,
		)}
	}
	if params.Proxy != nil {
		return InvalidError{Exception: fmt.Sprintf(
			"Proxy is not supported when %s=0 is set; unset it to use a proxy",
			controlPlaneSidecarCreateEnvVar,
		)}
	}
	return nil
}

func (s *sandboxImplV2) createSidecar(ctx context.Context, in sidecarCreateInputs) (sidecarCreateResult, error) {
	if !controlPlaneSidecarCreateEnabled() {
		return createSidecarViaTcr(ctx, s, in)
	}

	var workdir *string
	if in.params.Workdir != "" {
		workdir = &in.params.Workdir
	}

	var resources *pb.Resources
	if in.memoryReserveConsumeMib != nil {
		resources = pb.Resources_builder{MemoryMb: *in.memoryReserveConsumeMib}.Build()
	}

	req := pb.SandboxContainerCreateV2Request_builder{
		SandboxId:     s.id,
		ContainerName: in.name,
		Definition: pb.Sandbox_builder{
			ImageId:                  in.image.ImageID,
			EntrypointArgs:           in.params.Command,
			Workdir:                  workdir,
			SecretIds:                in.secretIds,
			VolumeMounts:             in.volumeMounts,
			CloudBucketMounts:        in.cloudBucketMounts,
			NetworkAccess:            in.networkAccess,
			IncludeOidcIdentityToken: in.params.IncludeOidcIdentityToken,
			ProxyId:                  in.proxyID,
			PtyInfo:                  in.ptyInfo,
			Resources:                resources,
		}.Build(),
		SecretSources: in.secretSources,
	}.Build()

	resp, err := s.client.cpClient.SandboxContainerCreateV2(ctx, req)
	if err != nil {
		return sidecarCreateResult{}, sidecarCreateError(err)
	}
	return sidecarCreateResult{containerID: resp.GetContainerId(), containerName: resp.GetContainerName()}, nil
}

// sbStdinV2 writes Sandbox stdin through the task command router.
type sbStdinV2 struct {
	getCommandRouter func(ctx context.Context) (string, *taskCommandRouterClient, error)

	mu     sync.Mutex
	offset uint64
}

func (s *sbStdinV2) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	taskID, client, err := s.getCommandRouter(context.Background())
	if err != nil {
		return 0, err
	}
	if err := client.SandboxStdinWriteV2(context.Background(), taskID, s.offset, p, false); err != nil {
		return 0, err
	}
	s.offset += uint64(len(p))
	return len(p), nil
}

func (s *sbStdinV2) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	taskID, client, err := s.getCommandRouter(context.Background())
	if err != nil {
		return err
	}
	return client.SandboxStdinWriteV2(context.Background(), taskID, s.offset, nil, true)
}

func outputStreamSbV2(ctx context.Context, getCommandRouter func(ctx context.Context) (string, *taskCommandRouterClient, error), fd pb.FileDescriptor) io.ReadCloser {
	taskID, commandRouterClient, err := getCommandRouter(ctx)
	if err != nil {
		return failedReader{err: fmt.Errorf("error getting output stream: %w", err)}
	}
	return commandRouterClient.SandboxStdioReadV2(ctx, taskID, fd)
}
