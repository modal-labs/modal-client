package modal

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/singleflight"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

// EnvironmentService looks up Modal Environments.
type EnvironmentService struct {
	// Objects provides methods for creating, listing, and deleting Environments.
	Objects EnvironmentObjectsService
	client  *Client
}

// EnvironmentFromNameParams are options for client.Environments.FromName.
type EnvironmentFromNameParams struct {
	CreateIfMissing bool
}

// Environment references a Modal Environment and exposes its resource namespaces.
type Environment struct {
	EnvironmentID string
	Name          string
	Apps          EnvironmentAppsService
}

// FromName looks up an Environment by name, optionally creating it if it does not exist.
func (s *EnvironmentService) FromName(ctx context.Context, name string, params *EnvironmentFromNameParams) (*Environment, error) {
	if err := checkEnvironmentName(name, true); err != nil {
		return nil, err
	}
	creationType := pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED
	if params != nil && params.CreateIfMissing {
		creationType = pb.ObjectCreationType_OBJECT_CREATION_TYPE_CREATE_IF_MISSING
	}
	resp, err := s.client.cpClient.EnvironmentGetOrCreate(ctx, pb.EnvironmentGetOrCreateRequest_builder{
		DeploymentName:     name,
		ObjectCreationType: creationType,
	}.Build())
	if status.Code(err) == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Environment '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}
	if metadata := resp.GetMetadata(); metadata != nil {
		name = metadata.GetName()
	}
	return newEnvironment(s.client, resp.GetEnvironmentId(), name), nil
}

func newEnvironment(client *Client, id, name string) *Environment {
	return &Environment{
		EnvironmentID: id,
		Name:          name,
		Apps:          &environmentAppsServiceImpl{client: client, environmentName: name},
	}
}

// EnvironmentObjectsService provides methods for managing Environment objects.
type EnvironmentObjectsService interface {
	Create(ctx context.Context, name string, params *EnvironmentCreateParams) error
	List(ctx context.Context) (iter.Seq2[*Environment, error], error)
	Delete(ctx context.Context, name string) error
}

type environmentObjectsServiceImpl struct{ client *Client }

// EnvironmentCreateParams are options for client.Environments.Objects.Create.
type EnvironmentCreateParams struct {
	// Restricted limits access to users with explicitly assigned Environment roles.
	Restricted bool
	// DefaultRole is the default Environment role for workspace members.
	DefaultRole string
}

// Create creates an Environment with the given name.
func (s *environmentObjectsServiceImpl) Create(ctx context.Context, name string, params *EnvironmentCreateParams) error {
	if err := checkEnvironmentName(name, false); err != nil {
		return err
	}
	if params == nil {
		params = &EnvironmentCreateParams{}
	}
	_, err := s.client.cpClient.EnvironmentCreate(ctx, pb.EnvironmentCreateRequest_builder{
		Name: name, IsManaged: params.Restricted, DefaultMemberRoleStr: params.DefaultRole,
	}.Build())
	switch status.Code(err) {
	case codes.AlreadyExists:
		return AlreadyExistsError{fmt.Sprintf("Environment '%s' already exists", name)}
	case codes.InvalidArgument:
		return InvalidError{status.Convert(err).Message()}
	}
	return err
}

// List yields Environment handles without additional lookup requests.
// RPC errors are yielded during iteration.
func (s *environmentObjectsServiceImpl) List(ctx context.Context) (iter.Seq2[*Environment, error], error) {
	return func(yield func(*Environment, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		resp, err := s.client.cpClient.EnvironmentList(ctx, &emptypb.Empty{})
		if err != nil {
			yield(nil, err)
			return
		}
		for _, item := range resp.GetItems() {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}
			if !yield(newEnvironment(s.client, item.GetEnvironmentId(), item.GetName()), nil) {
				return
			}
		}
	}, nil
}

// Delete permanently deletes an Environment and all objects in it.
func (s *environmentObjectsServiceImpl) Delete(ctx context.Context, name string) error {
	_, err := s.client.cpClient.EnvironmentDelete(ctx, pb.EnvironmentDeleteRequest_builder{Name: name}.Build())
	if status.Code(err) == codes.NotFound {
		return NotFoundError{fmt.Sprintf("Environment '%s' not found", name)}
	}
	return err
}

// EnvironmentAppsService accesses Apps in an Environment.
type EnvironmentAppsService interface {
	// List yields handles for live Apps, excluding stopped and disabled Apps.
	// Listing Apps does not create Apps. RPC errors are yielded during iteration.
	List(ctx context.Context) (iter.Seq2[*App, error], error)
}

type environmentAppsServiceImpl struct {
	client          *Client
	environmentName string
}

func (s *environmentAppsServiceImpl) List(ctx context.Context) (iter.Seq2[*App, error], error) {
	return func(yield func(*App, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		resp, err := s.client.cpClient.AppList(ctx, pb.AppListRequest_builder{
			EnvironmentName: s.environmentName,
		}.Build())
		if err != nil {
			yield(nil, err)
			return
		}
		for _, info := range resp.GetApps() {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}
			if info.GetState() == pb.AppState_APP_STATE_STOPPED || info.GetState() == pb.AppState_APP_STATE_DISABLED {
				continue
			}
			if !yield(newApp(s.client, info.GetAppId(), info.GetName(), s.environmentName, info.GetMetadata()), nil) {
				return
			}
		}
	}, nil
}

// fetchEnvironmentTimeout bounds a single environment fetch. The RPC runs on a
// context detached from the caller, so without this bound a hung server would
// block every concurrent waiter sharing the singleflight call indefinitely.
const fetchEnvironmentTimeout = 30 * time.Second

type environmentSettings struct {
	ImageBuilderVersion string
	WebhookSuffix       string
}

type environment struct {
	ID       string
	Name     string
	Settings environmentSettings
}

type environmentManager struct {
	client pb.ModalClientClient
	logger *slog.Logger

	mu    sync.RWMutex
	cache map[string]*environment
	group singleflight.Group
}

func newEnvironmentManager(client pb.ModalClientClient, logger *slog.Logger) *environmentManager {
	manager := &environmentManager{
		client: client,
		logger: logger,
		cache:  make(map[string]*environment),
	}
	return manager
}

// fetchEnvironment fetches the environment from the server and caches the result. Passing
// an empty name returns the default environment
func (manager *environmentManager) fetchEnvironment(ctx context.Context, name string) (*environment, error) {
	manager.mu.RLock()
	env, ok := manager.cache[name]
	manager.mu.RUnlock()
	if ok {
		return env, nil
	}

	ch := manager.group.DoChan(name, func() (any, error) {
		manager.mu.RLock()
		env, ok := manager.cache[name]
		manager.mu.RUnlock()
		if ok {
			return env, nil
		}
		// Run the RPC on a context detached from any individual caller so that
		// one caller leaving cannot fail the others sharing this flight. The
		// detached context is bounded by fetchEnvironmentTimeout rather than a
		// caller's deadline, keeping the flight's lifetime independent of which
		// caller happens to win it. Each caller still honors its own deadline
		// below by selecting on its ctx while waiting for the result.
		rpcCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), fetchEnvironmentTimeout)
		defer cancel()
		manager.logger.DebugContext(rpcCtx, "Fetching environment from server", "environment_name", name)

		resp, err := manager.client.EnvironmentGetOrCreate(rpcCtx, pb.EnvironmentGetOrCreateRequest_builder{
			DeploymentName: name,
		}.Build())
		if err != nil {
			return nil, err
		}

		metadata := resp.GetMetadata()
		settings := metadata.GetSettings()

		env = &environment{
			ID:   resp.GetEnvironmentId(),
			Name: metadata.GetName(),
			Settings: environmentSettings{
				ImageBuilderVersion: settings.GetImageBuilderVersion(),
				WebhookSuffix:       settings.GetWebhookSuffix(),
			},
		}

		manager.mu.Lock()
		manager.cache[name] = env
		manager.mu.Unlock()
		manager.logger.DebugContext(rpcCtx, "Cached environment",
			"environment_name", name,
			"environment_id", env.ID,
			"image_builder_version", env.Settings.ImageBuilderVersion,
		)

		return env, nil
	})

	// Wait for the shared flight, but honor this caller's own deadline or
	// cancellation. If ctx is done first, this caller returns early while the
	// flight keeps running for the remaining waiters and still populates the
	// cache for the next caller.
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-ch:
		if res.Err != nil {
			return nil, res.Err
		}
		return res.Val.(*environment), nil
	}
}

// GetImageBuilderVersion returns the image builder version
func (manager *environmentManager) GetImageBuilderVersion(ctx context.Context, environmentName string) (string, error) {
	env, err := manager.fetchEnvironment(ctx, environmentName)
	if err != nil {
		return "", fmt.Errorf("failed to get environment for image builder version: %w", err)
	}
	return env.Settings.ImageBuilderVersion, nil
}
