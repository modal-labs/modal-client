package modal

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// AppService provides App related operations.
type AppService interface {
	FromName(ctx context.Context, name string, params *AppFromNameParams) (*App, error)
}

type appServiceImpl struct{ client *Client }

// App references a deployed Modal App.
type App struct {
	AppID string
	Name  string
}

// parseGPUConfig parses a GPU configuration string into a GPUConfig object.
// The GPU string format is "type" or "type:count" (e.g. "T4", "A100:2").
// Returns an empty config if gpu is empty, or an error if the format is invalid.
func parseGPUConfig(gpu string) (*pb.GPUConfig, error) {
	if gpu == "" {
		return pb.GPUConfig_builder{}.Build(), nil
	}

	gpuType := gpu
	count := uint32(1)

	if strings.Contains(gpu, ":") {
		parts := strings.SplitN(gpu, ":", 2)
		gpuType = parts[0]
		parsedCount, err := strconv.ParseUint(parts[1], 10, 32)
		if err != nil || parsedCount < 1 {
			return nil, fmt.Errorf("invalid GPU count: %s, value must be a positive integer", parts[1])
		}
		count = uint32(parsedCount)
	}

	return pb.GPUConfig_builder{
		Type:    0, // Deprecated field, but required by proto
		Count:   count,
		GpuType: strings.ToUpper(gpuType),
	}.Build(), nil
}

// AppFromNameParams are options for client.Apps.FromName.
type AppFromNameParams struct {
	Environment     string
	CreateIfMissing bool
}

// FromName references an App with a given name, creating a new App if necessary.
func (s *appServiceImpl) FromName(ctx context.Context, name string, params *AppFromNameParams) (*App, error) {
	if params == nil {
		params = &AppFromNameParams{}
	}

	creationType := pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED
	if params.CreateIfMissing {
		creationType = pb.ObjectCreationType_OBJECT_CREATION_TYPE_CREATE_IF_MISSING
	}

	resp, err := s.client.cpClient.AppGetOrCreate(ctx, pb.AppGetOrCreateRequest_builder{
		AppName:            name,
		EnvironmentName:    environmentName(params.Environment, s.client.profile),
		ObjectCreationType: creationType,
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("App '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Retrieved App", "app_id", resp.GetAppId(), "app_name", name)
	return &App{AppID: resp.GetAppId(), Name: name}, nil
}
