package modal

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/singleflight"
)

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
