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
	taskSandbox
}

func newSandboxImplV1(client *Client, sandboxID string, opts taskSandboxOptions) *sandboxImplV1 {
	s := &sandboxImplV1{}
	s.init(client, sandboxID, s, opts)
	return s
}

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
	impl := newSandboxImpl(client, createResp.GetSandboxId(), taskSandboxOptions{})
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
	return newSandboxImpl(client, resp.GetSandboxId(), taskSandboxOptions{}), nil
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
				if !yield(newSandboxImpl(client, info.GetId(), taskSandboxOptions{}), nil) {
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

	impl := newSandboxImpl(client, resp.GetSandboxId(), taskSandboxOptions{})

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

func (s *sandboxImplV1) isV2() bool { return false }

func (s *sandboxImplV1) rpcWait(ctx context.Context, timeoutSecs float32) (*pb.SandboxWaitResponse, error) {
	return s.client.cpClient.SandboxWait(ctx, pb.SandboxWaitRequest_builder{SandboxId: s.id, Timeout: timeoutSecs}.Build())
}

func (s *sandboxImplV1) rpcTerminate(ctx context.Context) error {
	_, err := s.client.cpClient.SandboxTerminate(ctx, pb.SandboxTerminateRequest_builder{SandboxId: s.id}.Build())
	return err
}

func (s *sandboxImplV1) rpcGetTaskID(ctx context.Context) (*pb.SandboxGetTaskIdResponse, error) {
	return s.client.cpClient.SandboxGetTaskId(ctx, pb.SandboxGetTaskIdRequest_builder{SandboxId: s.id}.Build())
}

func (s *sandboxImplV1) rpcGetTunnels(ctx context.Context, timeout time.Duration) (*pb.SandboxGetTunnelsResponse, error) {
	return s.client.cpClient.SandboxGetTunnels(ctx, pb.SandboxGetTunnelsRequest_builder{
		SandboxId: s.id,
		Timeout:   float32(timeout.Seconds()),
	}.Build())
}

func (s *sandboxImplV1) rpcGetExitSnapshot(ctx context.Context, timeoutSecs float32, opts ...grpc.CallOption) (*pb.SandboxGetExitSnapshotResponse, error) {
	return s.client.cpClient.SandboxGetExitSnapshot(ctx, pb.SandboxGetExitSnapshotRequest_builder{
		SandboxId: s.id,
		Timeout:   timeoutSecs,
	}.Build(), opts...)
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
	return s.createSidecarViaCommandRouter(ctx, in)
}
