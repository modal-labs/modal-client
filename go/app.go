package modal

import (
	"context"
	"fmt"
	"maps"
	"strconv"
	"strings"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// AppService provides App related operations.
type AppService interface {
	FromName(ctx context.Context, name string, params *AppFromNameParams) (*App, error)
}

type appServiceImpl struct{ client *Client }

// App references a Modal App.
type App struct {
	AppID       string
	Name        string
	Environment string
	client      *Client
	metadataMu  sync.Mutex
	metadata    *pb.AppHandleMetadata
}

// AppState is the lifecycle state of a Modal App.
type AppState string

const (
	AppStateEphemeral    AppState = "ephemeral"
	AppStateDetached     AppState = "detached"
	AppStateDeployed     AppState = "deployed"
	AppStateStopping     AppState = "stopping"
	AppStateStopped      AppState = "stopped"
	AppStateInitializing AppState = "initializing"
	AppStateDisabled     AppState = "disabled"
	AppStateUnknown      AppState = "unknown"
)

// AppLifecycle contains timestamps and attributions for an App's lifecycle events.
// Optional fields are nil when the event or attribution does not apply.
type AppLifecycle struct {
	State      AppState
	Version    *int
	CreatedAt  time.Time
	CreatedBy  string
	DeployedAt *time.Time
	DeployedBy *string
	StoppedAt  *time.Time
	StoppedBy  *string
}

// AppInfo describes an App's lifecycle and member Functions and Servers.
type AppInfo struct {
	AppID       string
	Description string
	Lifecycle   AppLifecycle
	Functions   map[string]string // Function name to Function ID.
	Servers     map[string]string // Server name to Function ID.
}

// AppInfoParams are options for App.Info.
type AppInfoParams struct {
	// Refresh fetches current information instead of the cached snapshot.
	Refresh bool
}

func newApp(client *Client, id, name, environment string, metadata *pb.AppHandleMetadata) *App {
	return &App{AppID: id, Name: name, Environment: environment, client: client, metadata: metadata}
}

func appOptionalTime(timestamp float64) *time.Time {
	if timestamp <= 0 {
		return nil
	}
	value := unixTime(timestamp)
	return &value
}

func appOptionalString(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func appInfoFromProto(metadata *pb.AppHandleMetadata, id string) *AppInfo {
	lifecycle := metadata.GetLifecycle()
	state := AppStateUnknown
	switch lifecycle.GetAppState() {
	case pb.AppState_APP_STATE_UNSPECIFIED, pb.AppState_APP_STATE_EPHEMERAL:
		state = AppStateEphemeral
	case pb.AppState_APP_STATE_DETACHED, pb.AppState_APP_STATE_DETACHED_DISCONNECTED:
		state = AppStateDetached
	case pb.AppState_APP_STATE_DEPLOYED:
		state = AppStateDeployed
	case pb.AppState_APP_STATE_STOPPING:
		state = AppStateStopping
	case pb.AppState_APP_STATE_STOPPED:
		state = AppStateStopped
	case pb.AppState_APP_STATE_INITIALIZING:
		state = AppStateInitializing
	case pb.AppState_APP_STATE_DISABLED:
		state = AppStateDisabled
	}
	info := &AppInfo{
		AppID: id, Description: metadata.GetDescription(),
		Functions: make(map[string]string), Servers: make(map[string]string),
		Lifecycle: AppLifecycle{
			State: state, CreatedAt: unixTime(lifecycle.GetCreatedAt()), CreatedBy: lifecycle.GetCreatedBy(),
			DeployedAt: appOptionalTime(lifecycle.GetDeployedAt()), DeployedBy: appOptionalString(lifecycle.GetDeployedBy()),
			StoppedAt: appOptionalTime(lifecycle.GetStoppedAt()), StoppedBy: appOptionalString(lifecycle.GetStoppedBy()),
		},
	}
	maps.Copy(info.Functions, metadata.GetFunctions())
	maps.Copy(info.Servers, metadata.GetServers())
	if version := int(lifecycle.GetVersion()); version != 0 {
		info.Lifecycle.Version = &version
	}
	return info
}

// Info returns the snapshot from the previous lookup, listing, or refresh.
// Set Refresh to fetch current lifecycle information and member Function and Server IDs.
// The returned information is a copy that callers can modify independently.
func (a *App) Info(ctx context.Context, params *AppInfoParams) (*AppInfo, error) {
	a.metadataMu.Lock()
	defer a.metadataMu.Unlock()
	if a.metadata == nil || (params != nil && params.Refresh) {
		if a.client == nil {
			return nil, InvalidError{"App.Info requires a handle retrieved through a Client"}
		}
		resp, err := a.client.cpClient.AppGetInfo(ctx, pb.AppGetInfoRequest_builder{AppId: a.AppID}.Build())
		if status.Code(err) == codes.NotFound {
			return nil, NotFoundError{fmt.Sprintf("App '%s' not found", a.AppID)}
		}
		if err != nil {
			return nil, err
		}
		if resp.GetInfo() == nil {
			return nil, InvalidError{fmt.Sprintf("No information returned for App '%s'", a.AppID)}
		}
		a.metadata = resp.GetInfo()
	}
	return appInfoFromProto(a.metadata, a.AppID), nil
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
		EnvironmentName:    firstNonEmpty(params.Environment, s.client.profile.Environment),
		ObjectCreationType: creationType,
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("App '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Retrieved App", "app_id", resp.GetAppId(), "app_name", name)
	return newApp(s.client, resp.GetAppId(), name, firstNonEmpty(params.Environment, s.client.profile.Environment), resp.GetHandleMetadata()), nil
}

// AppStopParams are options for App.Stop.
type AppStopParams struct{}

// Stop stops the App.
//
// Stopping an App terminates all of its running containers and Sandboxes and
// prevents new inputs from being processed.
func (a *App) Stop(ctx context.Context, params *AppStopParams) error {
	if a.AppID == "" {
		return InvalidError{Exception: "App.Stop() cannot be called on an App without an ID."}
	}
	if a.client == nil {
		return InvalidError{Exception: "App.Stop() requires a handle retrieved through a Client."}
	}
	_, err := a.client.cpClient.AppStop(ctx, pb.AppStopRequest_builder{
		AppId:  a.AppID,
		Source: pb.AppStopSource_APP_STOP_SOURCE_PYTHON_CLIENT,
	}.Build())
	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return NotFoundError{fmt.Sprintf("App '%s' not found", a.AppID)}
	}
	return err
}
