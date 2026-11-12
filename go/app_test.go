package modal

import (
	"context"
	"errors"
	"iter"
	"sync"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type appInfoMockClient struct {
	pb.ModalClientClient
	lookup func(context.Context, *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error)
	info   func(context.Context, *pb.AppGetInfoRequest) (*pb.AppGetInfoResponse, error)
	stats  func(context.Context, *pb.AppGetTimeRangeStatsRequest) (*pb.AppGetTimeRangeStatsResponse, error)
}

func (m *appInfoMockClient) AppGetTimeRangeStats(ctx context.Context, req *pb.AppGetTimeRangeStatsRequest, _ ...grpc.CallOption) (*pb.AppGetTimeRangeStatsResponse, error) {
	return m.stats(ctx, req)
}

func TestAppStats(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	since := time.Date(2026, 11, 12, 10, 0, 0, 0, time.FixedZone("offset", 3600))
	until := since.Add(time.Hour)
	functions := pb.AppFunctionCounts_builder{Calls: pb.AppCallCounts_builder{Total: 10, Succeeded: 7, Failed: 3}.Build(), Containers: pb.AppContainerCounts_builder{ContainerErrors: 2, ContainerActiveCount: 5}.Build()}.Build()
	servers := pb.AppServerCounts_builder{Requests: pb.AppRequestCounts_builder{Total: 20, Ok: 16, Failed: 4}.Build(), Containers: pb.AppContainerCounts_builder{ContainerErrors: 1, ContainerActiveCount: 6}.Build()}.Build()
	deploymentCount := uint64(0)
	response := pb.AppGetTimeRangeStatsResponse_builder{
		Since: timestamppb.New(since), Until: timestamppb.New(until),
		Functions:       pb.AppFunctionStatsGroup_builder{Total: functions, ByName: map[string]*pb.AppFunctionStats{"embed": pb.AppFunctionStats_builder{ObjectId: "fu-embed", Counts: functions}.Build()}}.Build(),
		Servers:         pb.AppServerStatsGroup_builder{Total: servers, ByName: map[string]*pb.AppServerStats{"api": pb.AppServerStats_builder{ObjectId: "fu-api", Counts: servers}.Build()}}.Build(),
		Sandboxes:       pb.AppSandboxCounts_builder{Total: 9, Finished: 5, Live: 4}.Build(),
		Lifecycle:       pb.AppLifecycle_builder{AppState: pb.AppState_APP_STATE_STOPPED, CreatedAt: 1700000000, CreatedBy: "creator"}.Build(),
		DeploymentCount: &deploymentCount,
	}.Build()
	mock := &appInfoMockClient{stats: func(_ context.Context, req *pb.AppGetTimeRangeStatsRequest) (*pb.AppGetTimeRangeStatsResponse, error) {
		g.Expect(req.GetAppId()).To(gomega.Equal("ap-test"))
		g.Expect(req.GetSince().AsTime()).To(gomega.Equal(since.UTC()))
		g.Expect(req.GetUntil().AsTime()).To(gomega.Equal(until.UTC()))
		return response, nil
	}}
	app := newApp(newEnvironmentServiceTestClient(t, mock), "ap-test", "", "", nil)
	stats, err := app.Stats(t.Context(), &AppStatsParams{Since: &since, Until: &until})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	functionCounts := AppStatsFunctionCounts{Calls: AppStatsCallCounts{Total: 10, Succeeded: 7, Failed: 3}, Containers: AppStatsContainerCounts{Errors: 2, Total: 5}}
	serverCounts := AppStatsServerCounts{Requests: AppStatsRequestCounts{Total: 20, OK: 16, Failed: 4}, Containers: AppStatsContainerCounts{Errors: 1, Total: 6}}
	g.Expect(stats).To(gomega.Equal(&AppStats{
		Since: since.UTC(), Until: until.UTC(),
		Functions: AppStatsFunctionStatsGroup{Total: functionCounts, ByName: map[string]AppStatsFunctionStats{"embed": {ObjectID: "fu-embed", Counts: functionCounts}}},
		Servers:   AppStatsServerStatsGroup{Total: serverCounts, ByName: map[string]AppStatsServerStats{"api": {ObjectID: "fu-api", Counts: serverCounts}}},
		Sandboxes: AppStatsSandboxCounts{Total: 9, Finished: 5, Live: 4},
		Lifecycle: AppLifecycle{State: AppStateStopped, CreatedAt: time.Unix(1700000000, 0).UTC(), CreatedBy: "creator"}, DeploymentCount: &deploymentCount,
	}))
}

func TestAppStatsDefaultsAndErrors(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	before := time.Now().UTC()
	mock := &appInfoMockClient{stats: func(_ context.Context, req *pb.AppGetTimeRangeStatsRequest) (*pb.AppGetTimeRangeStatsResponse, error) {
		until := req.GetUntil().AsTime()
		g.Expect(until).To(gomega.BeTemporally(">=", before))
		g.Expect(until).To(gomega.BeTemporally("<=", time.Now().UTC()))
		g.Expect(until.Sub(req.GetSince().AsTime())).To(gomega.Equal(time.Hour))
		return pb.AppGetTimeRangeStatsResponse_builder{Since: req.GetSince(), Until: req.GetUntil()}.Build(), nil
	}}
	app := newApp(newEnvironmentServiceTestClient(t, mock), "ap-test", "", "", nil)
	stats, err := app.Stats(t.Context(), nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(stats.DeploymentCount).To(gomega.BeNil())
	g.Expect(stats.Functions.ByName).To(gomega.BeEmpty())
	g.Expect(stats.Servers.ByName).To(gomega.BeEmpty())
	until := time.Now()
	for _, since := range []time.Time{until, until.Add(time.Second)} {
		_, err := app.Stats(t.Context(), &AppStatsParams{Since: &since, Until: &until})
		g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
	}
	rpcErr := errors.New("stats unavailable")
	mock.stats = func(context.Context, *pb.AppGetTimeRangeStatsRequest) (*pb.AppGetTimeRangeStatsResponse, error) {
		return nil, rpcErr
	}
	_, err = app.Stats(t.Context(), nil)
	g.Expect(err).To(gomega.MatchError(rpcErr))
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

type sandboxStatsMockClient struct {
	pb.ModalClientClient
	handler func(context.Context, *pb.SandboxGetTimeRangeStatsRequest) (*pb.SandboxGetTimeRangeStatsResponse, error)
}

func (m *sandboxStatsMockClient) SandboxGetTimeRangeStats(ctx context.Context, req *pb.SandboxGetTimeRangeStatsRequest, _ ...grpc.CallOption) (*pb.SandboxGetTimeRangeStatsResponse, error) {
	return m.handler(ctx, req)
}

func TestAppSandboxesStats(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	since := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	until := since.Add(24 * time.Hour)
	var pages []string
	mock := &sandboxStatsMockClient{handler: func(ctx context.Context, req *pb.SandboxGetTimeRangeStatsRequest) (*pb.SandboxGetTimeRangeStatsResponse, error) {
		g.Expect(req.GetAppId()).To(gomega.Equal("ap-stats"))
		g.Expect(req.GetSince().AsTime()).To(gomega.Equal(since))
		g.Expect(req.GetUntil().AsTime()).To(gomega.Equal(until))
		g.Expect(req.GetGroupByTags()).To(gomega.Equal([]string{"team"}))
		g.Expect(req.GetTags()).To(gomega.Equal(map[string]string{"env": "prod"}))
		pages = append(pages, req.GetPageToken())
		tag := pb.SandboxStatsGroup_TagValue_builder{}.Build()
		next := "1"
		if req.GetPageToken() == "1" {
			tag = pb.SandboxStatsGroup_TagValue_builder{Value: proto.String("ml")}.Build()
			next = ""
		}
		group := pb.SandboxStatsGroup_builder{Tags: map[string]*pb.SandboxStatsGroup_TagValue{"team": tag}, CostStr: "0.12500000", CostByResourceStr: map[string]string{"CPU": "0.10000000", "Memory": "0.02500000"}}.Build()
		return pb.SandboxGetTimeRangeStatsResponse_builder{Groups: []*pb.SandboxStatsGroup{group}, NextPageToken: next}.Build(), nil
	}}
	app := newApp(&Client{cpClient: &clientWithConn{ModalClientClient: mock}}, "ap-stats", "", "", nil)
	groups, err := collectSandboxStats(app.Sandboxes.Stats(t.Context(), &SandboxStatsParams{Since: &since, Until: &until, GroupByTags: []string{"team"}, Tags: map[string]string{"env": "prod"}}))
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(pages).To(gomega.Equal([]string{"", "1"}))
	ml := "ml"
	g.Expect(groups).To(gomega.Equal([]SandboxGroupStats{
		{Tags: map[string]*string{"team": nil}, Cost: "0.12500000", CostByResource: map[string]string{"CPU": "0.10000000", "Memory": "0.02500000"}},
		{Tags: map[string]*string{"team": &ml}, Cost: "0.12500000", CostByResource: map[string]string{"CPU": "0.10000000", "Memory": "0.02500000"}},
	}))

	seq, err := app.Sandboxes.Stats(t.Context(), &SandboxStatsParams{Since: &since, Until: &until, GroupByTags: []string{"team"}, Tags: map[string]string{"env": "prod"}})
	g.Expect(err).NotTo(gomega.HaveOccurred())
	first, err := collectSandboxStats(seq, nil)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	again, err := collectSandboxStats(seq, nil)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(again).To(gomega.Equal(first))
	g.Expect(pages).To(gomega.Equal([]string{"", "1", "", "1", "", "1"}))

	mock.handler = func(context.Context, *pb.SandboxGetTimeRangeStatsRequest) (*pb.SandboxGetTimeRangeStatsResponse, error) {
		return nil, status.Error(codes.InvalidArgument, "Too many groups")
	}
	_, err = collectSandboxStats(app.Sandboxes.Stats(t.Context(), &SandboxStatsParams{Since: &since}))
	g.Expect(err).To(gomega.Equal(InvalidError{Exception: "Too many groups"}))

	future := time.Now().Add(time.Hour)
	for _, params := range []*SandboxStatsParams{nil, {}, {Since: &future}, {Since: &until, Until: &since}} {
		_, err := app.Sandboxes.Stats(t.Context(), params)
		g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
	}
	_, err = newApp(nil, "", "", "", nil).Sandboxes.Stats(t.Context(), &SandboxStatsParams{Since: &since})
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
	for _, m := range []*AppSandboxesManager{nil, {}} {
		_, err = m.Stats(t.Context(), &SandboxStatsParams{Since: &since})
		g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
	}
}

func collectSandboxStats(seq iter.Seq2[SandboxGroupStats, error], err error) ([]SandboxGroupStats, error) {
	if err != nil {
		return nil, err
	}
	var groups []SandboxGroupStats
	for group, err := range seq {
		if err != nil {
			return nil, err
		}
		groups = append(groups, group)
	}
	return groups, nil
}
