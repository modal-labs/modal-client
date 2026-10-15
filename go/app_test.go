package modal

import (
	"context"
	"sync"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type appInfoMockClient struct {
	pb.ModalClientClient
	lookup func(context.Context, *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error)
	info   func(context.Context, *pb.AppGetInfoRequest) (*pb.AppGetInfoResponse, error)
}

func (m *appInfoMockClient) AppGetOrCreate(ctx context.Context, req *pb.AppGetOrCreateRequest, _ ...grpc.CallOption) (*pb.AppGetOrCreateResponse, error) {
	return m.lookup(ctx, req)
}

func (m *appInfoMockClient) AppGetInfo(ctx context.Context, req *pb.AppGetInfoRequest, _ ...grpc.CallOption) (*pb.AppGetInfoResponse, error) {
	return m.info(ctx, req)
}

func TestAppInfoCachedAndRefresh(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	metadata := pb.AppHandleMetadata_builder{
		Description: "App description", Functions: map[string]string{"echo": "fu-echo"}, Servers: map[string]string{"api": "fu-api"},
		Lifecycle: pb.AppLifecycle_builder{
			AppState: pb.AppState_APP_STATE_DEPLOYED, Version: 3,
			CreatedAt: 1700000000.125, CreatedBy: "creator", DeployedAt: 1700000100, DeployedBy: "deployer",
		}.Build(),
	}.Build()
	mock := &appInfoMockClient{lookup: func(_ context.Context, req *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
		g.Expect(req.GetAppName()).To(gomega.Equal("libmodal-test"))
		return pb.AppGetOrCreateResponse_builder{AppId: "ap-test", HandleMetadata: metadata}.Build(), nil
	}}
	client := newEnvironmentServiceTestClient(t, mock)
	app, err := client.Apps.FromName(t.Context(), "libmodal-test", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	info, err := app.Info(t.Context(), nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.AppID).To(gomega.Equal("ap-test"))
	g.Expect(info.Description).To(gomega.Equal("App description"))
	g.Expect(info.Functions).To(gomega.Equal(map[string]string{"echo": "fu-echo"}))
	g.Expect(info.Servers).To(gomega.Equal(map[string]string{"api": "fu-api"}))
	g.Expect(info.Lifecycle.State).To(gomega.Equal(AppStateDeployed))
	g.Expect(info.Lifecycle.CreatedAt).To(gomega.Equal(time.Unix(1700000000, 125000000).UTC()))
	g.Expect(info.Lifecycle.CreatedBy).To(gomega.Equal("creator"))
	g.Expect(*info.Lifecycle.Version).To(gomega.Equal(3))
	g.Expect(*info.Lifecycle.DeployedAt).To(gomega.Equal(time.Unix(1700000100, 0).UTC()))
	g.Expect(*info.Lifecycle.DeployedBy).To(gomega.Equal("deployer"))
	g.Expect(info.Lifecycle.StoppedAt).To(gomega.BeNil())
	g.Expect(info.Lifecycle.StoppedBy).To(gomega.BeNil())
	info.Functions["echo"] = "modified"
	info.Servers["api"] = "modified"
	*info.Lifecycle.Version = 99
	*info.Lifecycle.DeployedAt = time.Time{}
	*info.Lifecycle.DeployedBy = "modified"
	info, err = app.Info(t.Context(), nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Functions["echo"]).To(gomega.Equal("fu-echo"))
	g.Expect(info.Servers["api"]).To(gomega.Equal("fu-api"))
	g.Expect(*info.Lifecycle.Version).To(gomega.Equal(3))
	g.Expect(*info.Lifecycle.DeployedAt).To(gomega.Equal(time.Unix(1700000100, 0).UTC()))
	g.Expect(*info.Lifecycle.DeployedBy).To(gomega.Equal("deployer"))
	mock.info = func(_ context.Context, req *pb.AppGetInfoRequest) (*pb.AppGetInfoResponse, error) {
		g.Expect(req.GetAppId()).To(gomega.Equal("ap-test"))
		return pb.AppGetInfoResponse_builder{Info: pb.AppHandleMetadata_builder{
			Description: "Refreshed", Lifecycle: pb.AppLifecycle_builder{
				AppState: pb.AppState_APP_STATE_STOPPED, StoppedAt: 1700000200, StoppedBy: "stopper",
			}.Build(),
		}.Build()}.Build(), nil
	}
	info, err = app.Info(t.Context(), &AppInfoParams{Refresh: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Description).To(gomega.Equal("Refreshed"))
	g.Expect(info.Lifecycle.State).To(gomega.Equal(AppStateStopped))
	g.Expect(*info.Lifecycle.StoppedAt).To(gomega.Equal(time.Unix(1700000200, 0).UTC()))
	g.Expect(*info.Lifecycle.StoppedBy).To(gomega.Equal("stopper"))
	other, err := app.Info(t.Context(), nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	*other.Lifecycle.StoppedAt = time.Time{}
	*other.Lifecycle.StoppedBy = "modified"
	for _, code := range []codes.Code{codes.PermissionDenied, codes.NotFound} {
		rpcErr := status.Error(code, "refresh failed")
		mock.info = func(context.Context, *pb.AppGetInfoRequest) (*pb.AppGetInfoResponse, error) { return nil, rpcErr }
		_, err = app.Info(t.Context(), &AppInfoParams{Refresh: true})
		if code == codes.NotFound {
			g.Expect(err).To(gomega.BeAssignableToTypeOf(NotFoundError{}))
		} else {
			g.Expect(err).To(gomega.Equal(rpcErr))
		}
		cached, err := app.Info(t.Context(), nil)
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		g.Expect(cached).To(gomega.Equal(info))
	}
}

func TestAppInfoConcurrentFetch(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	calls := 0
	mock := &appInfoMockClient{
		lookup: func(context.Context, *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
			return pb.AppGetOrCreateResponse_builder{AppId: "ap-test"}.Build(), nil
		},
		info: func(context.Context, *pb.AppGetInfoRequest) (*pb.AppGetInfoResponse, error) {
			calls++
			return pb.AppGetInfoResponse_builder{Info: pb.AppHandleMetadata_builder{Description: "Fetched"}.Build()}.Build(), nil
		},
	}
	client := newEnvironmentServiceTestClient(t, mock)
	app, err := client.Apps.FromName(t.Context(), "libmodal-test", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	results := make(chan *AppInfo, 8)
	errors := make(chan error, 8)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			info, err := app.Info(t.Context(), nil)
			results <- info
			errors <- err
		})
	}
	wg.Wait()
	close(results)
	close(errors)
	for err := range errors {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
	}
	for info := range results {
		g.Expect(info.Description).To(gomega.Equal("Fetched"))
	}
	g.Expect(calls).To(gomega.Equal(1))
}

func TestAppLifecycleStates(t *testing.T) {
	t.Parallel()
	for state, expected := range map[pb.AppState]AppState{
		pb.AppState_APP_STATE_UNSPECIFIED:           AppStateEphemeral,
		pb.AppState_APP_STATE_EPHEMERAL:             AppStateEphemeral,
		pb.AppState_APP_STATE_DETACHED:              AppStateDetached,
		pb.AppState_APP_STATE_DETACHED_DISCONNECTED: AppStateDetached,
		pb.AppState_APP_STATE_DEPLOYED:              AppStateDeployed,
		pb.AppState_APP_STATE_STOPPING:              AppStateStopping,
		pb.AppState_APP_STATE_STOPPED:               AppStateStopped,
		pb.AppState_APP_STATE_INITIALIZING:          AppStateInitializing,
		pb.AppState_APP_STATE_DISABLED:              AppStateDisabled,
		pb.AppState(999):                            AppStateUnknown,
	} {
		t.Run(string(expected)+state.String(), func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			info := appInfoFromProto(pb.AppHandleMetadata_builder{
				Lifecycle: pb.AppLifecycle_builder{AppState: state}.Build(),
			}.Build(), "ap-test")
			g.Expect(info.Lifecycle.State).To(gomega.Equal(expected))
			g.Expect(info.Lifecycle.Version).To(gomega.BeNil())
			g.Expect(info.Lifecycle.DeployedAt).To(gomega.BeNil())
			g.Expect(info.Lifecycle.DeployedBy).To(gomega.BeNil())
			g.Expect(info.Lifecycle.StoppedAt).To(gomega.BeNil())
			g.Expect(info.Lifecycle.StoppedBy).To(gomega.BeNil())
			g.Expect(info.Functions).NotTo(gomega.BeNil())
			g.Expect(info.Servers).NotTo(gomega.BeNil())
		})
	}
}

func TestParseGPUConfig(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	config, err := parseGPUConfig("")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config).ShouldNot(gomega.BeNil())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(0)))
	g.Expect(config.GetGpuType()).To(gomega.Equal(""))

	config, err = parseGPUConfig("T4")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(1)))
	g.Expect(config.GetGpuType()).To(gomega.Equal("T4"))

	config, err = parseGPUConfig("A10G")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(1)))
	g.Expect(config.GetGpuType()).To(gomega.Equal("A10G"))

	config, err = parseGPUConfig("A100-80GB")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(1)))
	g.Expect(config.GetGpuType()).To(gomega.Equal("A100-80GB"))

	config, err = parseGPUConfig("A100-80GB:3")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(3)))
	g.Expect(config.GetGpuType()).To(gomega.Equal("A100-80GB"))

	config, err = parseGPUConfig("T4:2")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(2)))
	g.Expect(config.GetGpuType()).To(gomega.Equal("T4"))

	config, err = parseGPUConfig("a100:4")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(config.GetCount()).To(gomega.Equal(uint32(4)))
	g.Expect(config.GetGpuType()).To(gomega.Equal("A100"))

	_, err = parseGPUConfig("T4:invalid")
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("invalid GPU count: invalid"))

	_, err = parseGPUConfig("T4:")
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("invalid GPU count: "))

	_, err = parseGPUConfig("T4:0")
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("invalid GPU count: 0"))

	_, err = parseGPUConfig("T4:-1")
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("invalid GPU count: -1"))
}
