package modal

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// mainContainerName is the reserved name of a Sandbox's main container.
const mainContainerName = "main"

// containerWaitPollTimeout is the per-RPC server-side wait timeout used while
// polling for a sidecar's terminal status.
const containerWaitPollTimeout = 10 * time.Second

// SidecarService creates and manages sidecar containers inside a Sandbox.
//
// EXPERIMENTAL: the API is subject to change.
type SidecarService interface {
	// Create starts a new sidecar container in the Sandbox. The Image must
	// already be built by calling [Image.Build] before it's passed to Create.
	Create(ctx context.Context, name string, image *Image, params *SidecarCreateParams) (*SidecarContainer, error)
	// Get returns a sidecar container by name.
	Get(ctx context.Context, name string, params *SidecarGetParams) (*SidecarContainer, error)
	// List returns all sidecar containers (not including the main container).
	List(ctx context.Context, params *SidecarListParams) ([]*SidecarContainer, error)
}

type sidecarServiceImpl struct{ sandbox *Sandbox }

// SidecarCreateParams holds options for creating a sidecar container.
type SidecarCreateParams struct {
	// Command to run in the sidecar container on startup.
	Command []string
	// Env are environment variables to set in the sidecar container.
	Env map[string]string
	// Secrets to inject into the sidecar container as environment variables.
	Secrets []*Secret
	// Workdir sets the working directory of the sidecar container.
	Workdir string
}

// SidecarGetParams holds options for retrieving a sidecar by name.
type SidecarGetParams struct {
	IncludeTerminated bool
}

// SidecarListParams holds options for listing sidecars.
type SidecarListParams struct {
	IncludeTerminated bool
}

// SidecarExecParams holds options for [SidecarContainer.Exec].
type SidecarExecParams SandboxExecParams

// SidecarWaitParams holds options for [SidecarContainer.Wait].
type SidecarWaitParams struct{}

// SidecarPollParams holds options for [SidecarContainer.Poll].
type SidecarPollParams struct{}

// SidecarTerminateParams holds options for [SidecarContainer.Terminate].
type SidecarTerminateParams struct {
	// Wait, when true, will wait for the sidecar container to terminate.
	Wait bool
}

func validateSidecarName(name string) error {
	if name == "" {
		return InvalidError{Exception: "sidecar name must not be empty"}
	}
	if name == mainContainerName {
		return InvalidError{Exception: fmt.Sprintf("the name %q is reserved for the Sandbox's main container. Use the Sandbox methods directly to interact with it", mainContainerName)}
	}
	return nil
}

func (s *sidecarServiceImpl) Create(ctx context.Context, name string, image *Image, params *SidecarCreateParams) (*SidecarContainer, error) {
	if err := validateSidecarName(name); err != nil {
		return nil, err
	}
	if image == nil || image.ImageID == "" {
		return nil, InvalidError{Exception: "sidecar image must already be built. Call image.Build(ctx, app) first or use Images.FromID(...)"}
	}
	if params == nil {
		params = &SidecarCreateParams{}
	}
	if err := validateWorkdir(params.Workdir); err != nil {
		return nil, err
	}
	if err := validateExecArgs(params.Command); err != nil {
		return nil, err
	}

	// Locally-created Secrets (FromMap) are passed directly to the worker as
	// environment variables, so only the remaining Secrets need hydrating. This
	// avoids a SecretGetOrCreate round-trip for env-dict Secrets.
	envDict, resolvableSecrets := splitEnvDictAndResolvableSecrets(params.Secrets)
	for k, v := range params.Env {
		if err := validateEnvVarName(k); err != nil {
			return nil, err
		}
		envDict[k] = v
	}
	if err := hydrateSecrets(ctx, s.sandbox.client, resolvableSecrets); err != nil {
		return nil, err
	}
	secretIds, err := collectSecretIDs(resolvableSecrets)
	if err != nil {
		return nil, err
	}

	taskID, client, err := s.sandbox.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	req := pb.TaskContainerCreateRequest_builder{
		TaskId:        taskID,
		ContainerName: name,
		ImageId:       image.ImageID,
		Args:          params.Command,
		Env:           envDict,
		Workdir:       params.Workdir,
		SecretIds:     secretIds,
	}.Build()

	resp, err := client.ContainerCreate(ctx, req)
	if err != nil {
		if st, ok := status.FromError(err); ok {
			switch st.Code() {
			case codes.AlreadyExists:
				return nil, AlreadyExistsError{Exception: st.Message()}
			case codes.InvalidArgument:
				return nil, InvalidError{Exception: st.Message()}
			}
		}
		return nil, err
	}

	containerName := resp.GetContainerName()
	if containerName == "" {
		containerName = name
	}
	s.sandbox.client.logger.DebugContext(ctx, "Created SidecarContainer",
		"container_id", resp.GetContainerId(),
		"container_name", containerName,
		"sandbox_id", s.sandbox.SandboxID)
	return newSidecarContainer(s.sandbox, resp.GetContainerId(), containerName, nil), nil
}

func (s *sidecarServiceImpl) Get(ctx context.Context, name string, params *SidecarGetParams) (*SidecarContainer, error) {
	if err := validateSidecarName(name); err != nil {
		return nil, err
	}
	if params == nil {
		params = &SidecarGetParams{}
	}

	taskID, client, err := s.sandbox.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	resp, err := client.ContainerGet(ctx, pb.TaskContainerGetRequest_builder{
		TaskId:            taskID,
		ContainerName:     name,
		IncludeTerminated: params.IncludeTerminated,
	}.Build())
	if err != nil {
		if st, ok := status.FromError(err); ok && st.Code() == codes.NotFound {
			return nil, NotFoundError{Exception: fmt.Sprintf("Sidecar container %q not found", name)}
		}
		return nil, err
	}
	container := resp.GetContainer()
	if container == nil {
		return nil, ExecutionError{Exception: fmt.Sprintf("server returned no container for sidecar %q", name)}
	}
	return sidecarContainerFromProto(s.sandbox, container), nil
}

func (s *sidecarServiceImpl) List(ctx context.Context, params *SidecarListParams) ([]*SidecarContainer, error) {
	if params == nil {
		params = &SidecarListParams{}
	}

	taskID, client, err := s.sandbox.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	resp, err := client.ContainerList(ctx, pb.TaskContainerListRequest_builder{
		TaskId:            taskID,
		IncludeTerminated: params.IncludeTerminated,
	}.Build())
	if err != nil {
		return nil, err
	}

	containers := resp.GetContainers()
	out := make([]*SidecarContainer, 0, len(containers))
	for _, info := range containers {
		if info.GetContainerName() == mainContainerName {
			continue
		}
		out = append(out, sidecarContainerFromProto(s.sandbox, info))
	}
	return out, nil
}

// SidecarContainer is a handle to a sidecar container running in a Sandbox.
type SidecarContainer struct {
	// ContainerID is the fully qualified container ID.
	ContainerID string
	// ContainerName is the logical name of the container within the Sandbox.
	ContainerName string

	// Filesystem provides high-level filesystem operations for this container.
	Filesystem *SandboxFilesystem

	sandbox *Sandbox

	resultMu sync.Mutex
	result   *pb.GenericResult
}

func newSidecarContainer(sandbox *Sandbox, id, name string, result *pb.GenericResult) *SidecarContainer {
	c := &SidecarContainer{
		ContainerID:   id,
		ContainerName: name,
		sandbox:       sandbox,
		result:        result,
	}
	c.Filesystem = &SandboxFilesystem{sandbox: c, logger: sandbox.client.logger}
	return c
}

func sidecarContainerFromProto(sandbox *Sandbox, info *pb.TaskContainerInfo) *SidecarContainer {
	if info == nil {
		return nil
	}
	var result *pb.GenericResult
	if info.HasResult() {
		result = info.GetResult()
	}
	return newSidecarContainer(sandbox, info.GetContainerId(), info.GetContainerName(), result)
}

// Exec runs a command in the sidecar container and returns the process handle.
func (c *SidecarContainer) Exec(ctx context.Context, command []string, params *SidecarExecParams) (*ContainerProcess, error) {
	return c.sandbox.execInternal(ctx, command, (*SandboxExecParams)(params), c.ContainerID)
}

// execForFilesystem satisfies the sandboxForFilesystem interface for SidecarContainer.
func (c *SidecarContainer) execForFilesystem(ctx context.Context, command []string, params *SandboxExecParams) (*ContainerProcess, error) {
	return c.sandbox.execInternal(ctx, command, params, c.ContainerID)
}

// Wait blocks until the sidecar container exits, and returns its exit code.
func (c *SidecarContainer) Wait(ctx context.Context, _ *SidecarWaitParams) (int, error) {
	c.resultMu.Lock()
	cached := c.result
	c.resultMu.Unlock()
	if code := getReturnCode(cached); code != nil {
		return *code, nil
	}

	taskID, client, err := c.sandbox.getCommandRouter(ctx)
	if err != nil {
		return 0, err
	}

	for {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		resp, err := client.ContainerWait(ctx, pb.TaskContainerWaitRequest_builder{
			TaskId:      taskID,
			ContainerId: c.ContainerID,
			Timeout:     float32(containerWaitPollTimeout.Seconds()),
		}.Build())
		if err != nil {
			return 0, err
		}
		result := resp.GetResult()
		if result == nil || result.GetStatus() == pb.GenericResult_GENERIC_STATUS_UNSPECIFIED {
			continue
		}
		c.resultMu.Lock()
		c.result = result
		c.resultMu.Unlock()
		if code := getReturnCode(result); code != nil {
			return *code, nil
		}
		return 0, nil
	}
}

// Poll checks if the sidecar container has finished running.
// Returns nil if the container is still running, else returns the exit code.
func (c *SidecarContainer) Poll(ctx context.Context, _ *SidecarPollParams) (*int, error) {
	c.resultMu.Lock()
	cached := c.result
	c.resultMu.Unlock()
	if code := getReturnCode(cached); code != nil {
		return code, nil
	}

	taskID, client, err := c.sandbox.getCommandRouter(ctx)
	if err != nil {
		return nil, err
	}

	resp, err := client.ContainerWait(ctx, pb.TaskContainerWaitRequest_builder{
		TaskId:      taskID,
		ContainerId: c.ContainerID,
		Timeout:     0,
	}.Build())
	if err != nil {
		return nil, err
	}
	result := resp.GetResult()
	if result == nil {
		return nil, nil
	}
	if result.GetStatus() != pb.GenericResult_GENERIC_STATUS_UNSPECIFIED {
		c.resultMu.Lock()
		c.result = result
		c.resultMu.Unlock()
	}
	return getReturnCode(result), nil
}

// Terminate stops the sidecar container.
//
// The returned exit code is only meaningful when Wait is true.
func (c *SidecarContainer) Terminate(ctx context.Context, params *SidecarTerminateParams) (int, error) {
	if params == nil {
		params = &SidecarTerminateParams{}
	}

	taskID, client, err := c.sandbox.getCommandRouter(ctx)
	if err != nil {
		return 0, err
	}

	if err := client.ContainerTerminate(ctx, pb.TaskContainerTerminateRequest_builder{
		TaskId:      taskID,
		ContainerId: c.ContainerID,
	}.Build()); err != nil {
		return 0, err
	}

	if !params.Wait {
		return 0, nil
	}
	return c.Wait(ctx, nil)
}
