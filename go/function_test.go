package modal

import (
	"context"
	"errors"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestShouldUpload(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	const maxObjectSize = 2 * 1024 * 1024 // 2 MiB
	const maxAsyncObjectSize = 8 * 1024   // 8 KiB
	sync := pb.FunctionCallInvocationType_FUNCTION_CALL_INVOCATION_TYPE_SYNC
	async := pb.FunctionCallInvocationType_FUNCTION_CALL_INVOCATION_TYPE_ASYNC

	// Sync invocations only use the sync threshold, even above the async threshold.
	g.Expect(shouldUpload(maxAsyncObjectSize+1, maxObjectSize, maxAsyncObjectSize, sync)).To(gomega.BeFalse())
	// Exactly at the threshold should not upload (strict greater-than).
	g.Expect(shouldUpload(maxObjectSize, maxObjectSize, maxAsyncObjectSize, sync)).To(gomega.BeFalse())
	// Above the sync threshold should upload.
	g.Expect(shouldUpload(maxObjectSize+1, maxObjectSize, maxAsyncObjectSize, sync)).To(gomega.BeTrue())

	// Async invocations use the smaller async threshold.
	g.Expect(shouldUpload(maxAsyncObjectSize, maxObjectSize, maxAsyncObjectSize, async)).To(gomega.BeFalse())
	g.Expect(shouldUpload(maxAsyncObjectSize+1, maxObjectSize, maxAsyncObjectSize, async)).To(gomega.BeTrue())
	g.Expect(shouldUpload(maxObjectSize+1, maxObjectSize, maxAsyncObjectSize, async)).To(gomega.BeTrue())
}

func TestFunctionWithOptions(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		t.Fatalf("Failed to get Function: %v", err)
	}

	cpu := 2.0
	cpuLimit := 4.5
	routingRegion := "us-east"

	echoWithOptions := echo.WithOptions(&FunctionWithOptionsParams{
		CPU:           &cpu,
		CPULimit:      &cpuLimit,
		RoutingRegion: &routingRegion,
	})

	g.Expect(echoWithOptions.options).To(gomega.Equal(&functionOptions{
		cpu:           &cpu,
		cpuLimit:      &cpuLimit,
		routingRegion: &routingRegion,
	}))
}

func TestFunctionWithConcurrency(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		t.Fatalf("Failed to get Function: %v", err)
	}

	params := FunctionWithConcurrencyParams{
		MaxInputs: 10,
	}

	echoWithOptions := echo.WithConcurrency(&params)

	g.Expect(echoWithOptions.options).To(gomega.Equal(&functionOptions{
		maxConcurrentInputs: &params.MaxInputs,
	}))
}

func TestFunctionWithBatching(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		t.Fatalf("Failed to get Function: %v", err)
	}

	params := FunctionWithBatchingParams{
		MaxBatchSize: 10,
		Wait:         10 * time.Second,
	}

	echoWithOptions := echo.WithBatching(&params)

	g.Expect(echoWithOptions.options).To(gomega.Equal(&functionOptions{
		batchMaxSize: &params.MaxBatchSize,
		batchWait:    &params.Wait,
	}))
}

func TestFunctionWithOptionsSuccessive(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		t.Fatalf("Failed to get Function: %v", err)
	}

	cpu := 2.0
	cpuLimit := 4.5

	echoWithOptions := echo.
		WithOptions(&FunctionWithOptionsParams{CPU: &cpu}).
		WithOptions(&FunctionWithOptionsParams{CPULimit: &cpuLimit})

	g.Expect(echoWithOptions.options).To(gomega.Equal(&functionOptions{
		cpu:      &cpu,
		cpuLimit: &cpuLimit,
	}))
}

func TestDynamicFunctionConfigurationE2E(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		t.Fatalf("Failed to get Function: %v", err)
	}

	cpu := 2.0
	cpuLimit := 4.5
	options := FunctionWithOptionsParams{
		CPU:      &cpu,
		CPULimit: &cpuLimit,
	}

	concurrency := FunctionWithConcurrencyParams{
		MaxInputs: 10,
	}

	batching := FunctionWithBatchingParams{
		MaxBatchSize: 10,
		Wait:         10 * time.Second,
	}

	configured := echo.WithOptions(&options).WithConcurrency(&concurrency).WithBatching(&batching)

	g.Expect(configured.options).To(gomega.Equal(
		&functionOptions{
			cpu:      &cpu,
			cpuLimit: &cpuLimit,

			maxConcurrentInputs: &concurrency.MaxInputs,

			batchMaxSize: &batching.MaxBatchSize,
			batchWait:    &batching.Wait,
		},
	))

	g.Expect(&echo).ToNot(gomega.Equal(&configured))
	g.Expect(echo.options).To(gomega.Equal(&functionOptions{}))
}

func TestInstance(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		t.Fatalf("Failed to get Function: %v", err)
	}

	cpu := 2.0

	configuredEcho, err := echo.
		WithOptions(&FunctionWithOptionsParams{CPU: &cpu}).
		WithBatching(&FunctionWithBatchingParams{MaxBatchSize: 10}).
		WithConcurrency(&FunctionWithConcurrencyParams{MaxInputs: 10}).
		Instance(ctx)

	g.Expect(err).To(gomega.BeNil())
	g.Expect(configuredEcho.FunctionID).To(gomega.Not(gomega.BeEquivalentTo(echo.FunctionID)))
}

type statsMockClient struct {
	pb.ModalClientClient
	handler func(context.Context, *pb.FunctionGetTimeRangeStatsRequest) (*pb.FunctionGetTimeRangeStatsResponse, error)
}

func (m *statsMockClient) FunctionGetTimeRangeStats(ctx context.Context, req *pb.FunctionGetTimeRangeStatsRequest, _ ...grpc.CallOption) (*pb.FunctionGetTimeRangeStatsResponse, error) {
	return m.handler(ctx, req)
}

func TestFunctionStats(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	since := time.Date(2026, 9, 1, 0, 0, 0, 123000000, time.UTC)
	until := since.Add(time.Hour)
	distribution := pb.StatsPercentileDistribution_builder{Unit: "seconds", Percentiles: []*pb.StatsPercentile{pb.StatsPercentile_builder{PercentileBasisPoints: 9990, Value: 1.5}.Build()}}.Build()
	mock := &statsMockClient{handler: func(ctx context.Context, req *pb.FunctionGetTimeRangeStatsRequest) (*pb.FunctionGetTimeRangeStatsResponse, error) {
		g.Expect(ctx).To(gomega.Equal(t.Context()))
		g.Expect(req.GetFunctionId()).To(gomega.Equal("fu-stats"))
		g.Expect(req.GetSince().AsTime()).To(gomega.Equal(since))
		g.Expect(req.GetUntil().AsTime()).To(gomega.Equal(until))
		g.Expect(req.GetContainerId()).To(gomega.Equal("ta-container"))
		g.Expect(req.GetRollup()).To(gomega.BeTrue())
		return pb.FunctionGetTimeRangeStatsResponse_builder{Since: timestamppb.New(since), Until: timestamppb.New(until), InputSuccessCount: 10, InputFailureCount: 2, InputTimeoutCount: 3, InputRunningAtEndCount: 4, ContainerStartedCount: 5, ContainerErrorCount: 6, ContainerCreatingAtEndCount: 7, VariantCount: 8, InputPercentileStats: map[string]*pb.StatsPercentileDistribution{"execution_time": distribution}, ContainerPercentileStats: map[string]*pb.StatsPercentileDistribution{"custom_metric": distribution}}.Build(), nil
	}}
	f := &Function{FunctionID: "fu-stats", client: &Client{cpClient: &clientWithConn{ModalClientClient: mock}}}
	result, err := f.Stats(t.Context(), &FunctionStatsParams{Since: &since, Until: &until, Container: "ta-container", AllVariants: true})
	g.Expect(err).NotTo(gomega.HaveOccurred())
	converted := StatsPercentileDistribution{Unit: "seconds", Percentiles: []StatsPercentile{{Percentile: 99.9, Value: 1.5}}}
	g.Expect(result).To(gomega.Equal(&FunctionStats{Since: since, Until: until, InputSuccessCount: 10, InputFailureCount: 2, InputTimeoutCount: 3, InputRunningAtEndCount: 4, ContainerStartedCount: 5, ContainerErrorCount: 6, ContainerCreatingAtEndCount: 7, VariantCount: 8, AllVariants: true, InputPercentileStats: map[string]StatsPercentileDistribution{"execution_time": converted}, ContainerPercentileStats: map[string]StatsPercentileDistribution{"custom_metric": converted}}))
}

func TestFunctionStatsDefaultsAndErrors(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	until := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	for _, params := range []*FunctionStatsParams{nil, {Until: &until}} {
		before := time.Now()
		mock := &statsMockClient{handler: func(_ context.Context, req *pb.FunctionGetTimeRangeStatsRequest) (*pb.FunctionGetTimeRangeStatsResponse, error) {
			g.Expect(req.GetUntil().AsTime().Sub(req.GetSince().AsTime())).To(gomega.Equal(time.Hour))
			g.Expect(req.HasContainerId()).To(gomega.BeFalse())
			g.Expect(req.GetRollup()).To(gomega.BeFalse())
			if params == nil {
				g.Expect(req.GetUntil().AsTime().Before(before)).To(gomega.BeFalse())
			} else {
				g.Expect(req.GetUntil().AsTime()).To(gomega.Equal(until))
			}
			return pb.FunctionGetTimeRangeStatsResponse_builder{Since: req.GetSince(), Until: req.GetUntil()}.Build(), nil
		}}
		f := &Function{client: &Client{cpClient: &clientWithConn{ModalClientClient: mock}}}
		result, err := f.Stats(t.Context(), params)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(result.InputPercentileStats).To(gomega.BeEmpty())
		g.Expect(result.ContainerPercentileStats).To(gomega.BeEmpty())
		g.Expect(result.InputSuccessCount).To(gomega.BeZero())
		g.Expect(result.AllVariants).To(gomega.BeFalse())
	}
	f := &Function{}
	for _, since := range []time.Time{until, until.Add(time.Second)} {
		_, err := f.Stats(t.Context(), &FunctionStatsParams{Since: &since, Until: &until})
		g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
	}
	rpcErr := errors.New("stats unavailable")
	f.client = &Client{cpClient: &clientWithConn{ModalClientClient: &statsMockClient{handler: func(context.Context, *pb.FunctionGetTimeRangeStatsRequest) (*pb.FunctionGetTimeRangeStatsResponse, error) {
		return nil, rpcErr
	}}}}
	_, err := f.Stats(t.Context(), nil)
	g.Expect(err).To(gomega.MatchError(rpcErr))
}

func TestFunctionInfo(t *testing.T) {
	g := gomega.NewWithT(t)

	ctx := context.Background()
	mc, err := NewClient()
	g.Expect(err).To(gomega.BeNil())

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	g.Expect(err).To(gomega.BeNil())

	info, err := echo.Info(ctx, nil)
	g.Expect(err).To(gomega.BeNil())

	// Hack so we don't have to hardcode the Image ID
	g.Expect(info.ImageID).ToNot(gomega.BeEmpty())
	info.ImageID = ""

	fiveMinutes := 5 * time.Minute
	g.Expect(*info).To(gomega.BeEquivalentTo(FunctionInfo{
		StartupTimeout: &fiveMinutes,
		Timeout:        fiveMinutes,
		GPUs: []struct {
			GpuType string
			Count   uint32
		}{},
		Volumes: map[string]struct {
			VolumeID string
			ReadOnly bool
			SubPath  *string
		}{},
		CloudBucketMounts: map[string]struct {
			BucketName string
			BucketType string
			ReadOnly   bool
			KeyPrefix  *string
		}{},
	}))

	CPU := 2.0
	CPULimit := 3.0
	MemoryMiB := 512
	MemoryLimitMiB := 1024
	GPU := "H100:8"

	sec, err := mc.Secrets.FromName(ctx, "libmodal-test-secret", nil)
	g.Expect(err).To(gomega.BeNil())

	Secrets := []*Secret{sec}

	vol, err := mc.Volumes.FromName(ctx, "libmodal-example-volume", nil)
	g.Expect(err).To(gomega.BeNil())

	tru := true
	Volumes := map[string]*Volume{"/mnt/tmp": vol.WithMountOptions(&VolumeMountOptionsParams{&tru, nil})}
	MaxContainers := 10
	BufferContainers := 2
	ScaledownWindow := 15 * time.Second
	Timeout := 12 * time.Hour
	RoutingRegion := "us-west"

	unhydrated := echo.WithOptions(&FunctionWithOptionsParams{
		&CPU,
		&CPULimit,
		&MemoryMiB,
		&MemoryLimitMiB,
		&GPU,
		map[string]string{"ENV": "true"},
		Secrets,
		Volumes,
		nil,
		&MaxContainers,
		&BufferContainers,
		&ScaledownWindow,
		&Timeout,
		&RoutingRegion,
	}).
		WithBatching(&FunctionWithBatchingParams{10, 60 * time.Second}).
		WithConcurrency(&FunctionWithConcurrencyParams{MaxInputs: 10})

	unhydratedInfo, err := unhydrated.Info(ctx, nil)
	g.Expect(err).To(gomega.BeNil())

	g.Expect(*unhydratedInfo.CPU).To(gomega.BeEquivalentTo(2))
	g.Expect(*unhydratedInfo.CPULimit).To(gomega.BeEquivalentTo(3))
	g.Expect(*unhydratedInfo.MemoryMiB).To(gomega.BeEquivalentTo(512))
	g.Expect(*unhydratedInfo.MemoryLimitMiB).To(gomega.BeEquivalentTo(1024))
	g.Expect(unhydratedInfo.GPUs).To(gomega.BeEquivalentTo([]struct {
		GpuType string
		Count   uint32
	}{{GpuType: "H100", Count: 8}}))
	g.Expect(unhydratedInfo.EphemeralDiskMiB).To(gomega.BeNil())
	g.Expect(*unhydratedInfo.StartupTimeout).To(gomega.BeEquivalentTo(fiveMinutes))
	g.Expect(unhydratedInfo.Timeout).To(gomega.BeEquivalentTo(12 * time.Hour))
	g.Expect(unhydratedInfo.MaxRetries).To(gomega.BeNil())
	g.Expect(unhydratedInfo.Nonpreemptible).To(gomega.BeEquivalentTo(false))
	g.Expect(unhydratedInfo.Regions).To(gomega.BeNil())
	g.Expect(*unhydratedInfo.RoutingRegion).To(gomega.BeEquivalentTo("us-west"))
	g.Expect(unhydratedInfo.Cloud).To(gomega.BeNil())
	g.Expect(unhydratedInfo.ClusterInfo).To(gomega.BeNil())
	g.Expect(*unhydratedInfo.BatchingInfo).To(gomega.BeEquivalentTo(struct {
		MaxBatchSize uint32
		WaitMs       uint64
	}{MaxBatchSize: 10, WaitMs: 60000}))

	ten := uint32(10)
	g.Expect(*unhydratedInfo.ConcurrencyInfo).To(gomega.BeEquivalentTo(struct {
		MaxInputs    *uint32
		TargetInputs *uint32
	}{MaxInputs: &ten, TargetInputs: nil}))
	g.Expect(unhydratedInfo.Schedule).To(gomega.BeNil())
	g.Expect(unhydratedInfo.RestrictModalAccess).To(gomega.BeEquivalentTo(false))
	g.Expect(unhydratedInfo.BlockNetwork).To(gomega.BeEquivalentTo(false))
	g.Expect(unhydratedInfo.SingleUseContainers).To(gomega.BeEquivalentTo(false))
	g.Expect(unhydratedInfo.Volumes).To(gomega.BeEquivalentTo(
		map[string]struct {
			VolumeID string
			ReadOnly bool
			SubPath  *string
		}{
			"/mnt/tmp": {
				VolumeID: vol.VolumeID,
				ReadOnly: true,
				SubPath:  nil,
			},
		},
	))

	g.Expect(unhydratedInfo.CloudBucketMounts).To(gomega.BeEmpty())
	g.Expect(len(unhydratedInfo.SecretIDs)).To(gomega.Equal(1))
	g.Expect(unhydratedInfo.WebInfo).To(gomega.BeNil())
	g.Expect(unhydratedInfo.MethodNames).To(gomega.BeNil())
	g.Expect(unhydratedInfo.MethodDetails).To(gomega.BeNil())

	hydrated, err := unhydrated.Instance(ctx)
	g.Expect(err).To(gomega.BeNil())

	hydratedInfo, err := hydrated.Info(ctx, nil)
	g.Expect(err).To(gomega.BeNil())

	g.Expect(*hydratedInfo.CPU).To(gomega.BeEquivalentTo(*unhydratedInfo.CPU))
	g.Expect(*hydratedInfo.CPULimit).To(gomega.BeEquivalentTo(*unhydratedInfo.CPULimit))
	g.Expect(*hydratedInfo.MemoryMiB).To(gomega.BeEquivalentTo(*unhydratedInfo.MemoryMiB))
	g.Expect(*hydratedInfo.MemoryLimitMiB).To(gomega.BeEquivalentTo(*unhydratedInfo.MemoryLimitMiB))
	g.Expect(hydratedInfo.GPUs).To(gomega.BeEquivalentTo(unhydratedInfo.GPUs))
	g.Expect(hydratedInfo.EphemeralDiskMiB).To(gomega.BeEquivalentTo(unhydratedInfo.EphemeralDiskMiB))
	g.Expect(*hydratedInfo.StartupTimeout).To(gomega.BeEquivalentTo(*unhydratedInfo.StartupTimeout))
	g.Expect(hydratedInfo.Timeout).To(gomega.BeEquivalentTo(unhydratedInfo.Timeout))
	g.Expect(hydratedInfo.MaxRetries).To(gomega.BeEquivalentTo(unhydratedInfo.MaxRetries))
	g.Expect(hydratedInfo.Nonpreemptible).To(gomega.BeEquivalentTo(unhydratedInfo.Nonpreemptible))
	g.Expect(hydratedInfo.Regions).To(gomega.BeEquivalentTo(unhydratedInfo.Regions))
	g.Expect(*hydratedInfo.RoutingRegion).To(gomega.BeEquivalentTo(*unhydratedInfo.RoutingRegion))
	g.Expect(hydratedInfo.Cloud).To(gomega.BeEquivalentTo(unhydratedInfo.Cloud))
	g.Expect(hydratedInfo.ClusterInfo).To(gomega.BeEquivalentTo(unhydratedInfo.ClusterInfo))
	g.Expect(*hydratedInfo.BatchingInfo).To(gomega.BeEquivalentTo(*unhydratedInfo.BatchingInfo))
	g.Expect(*hydratedInfo.ConcurrencyInfo).To(gomega.BeEquivalentTo(*unhydratedInfo.ConcurrencyInfo))
	g.Expect(hydratedInfo.Schedule).To(gomega.BeEquivalentTo(unhydratedInfo.Schedule))
	g.Expect(hydratedInfo.RestrictModalAccess).To(gomega.BeEquivalentTo(unhydratedInfo.RestrictModalAccess))
	g.Expect(hydratedInfo.BlockNetwork).To(gomega.BeEquivalentTo(unhydratedInfo.BlockNetwork))
	g.Expect(hydratedInfo.SingleUseContainers).To(gomega.BeEquivalentTo(unhydratedInfo.SingleUseContainers))
	g.Expect(hydratedInfo.Volumes).To(gomega.BeEquivalentTo(unhydratedInfo.Volumes))
	g.Expect(hydratedInfo.CloudBucketMounts).To(gomega.BeEquivalentTo(unhydratedInfo.CloudBucketMounts))
	g.Expect(hydratedInfo.WebInfo).To(gomega.BeEquivalentTo(unhydratedInfo.WebInfo))
	g.Expect(hydratedInfo.MethodNames).To(gomega.BeEquivalentTo(unhydratedInfo.MethodNames))
	g.Expect(hydratedInfo.MethodDetails).To(gomega.BeEquivalentTo(unhydratedInfo.MethodDetails))

	// The Env has been hydrated into a secret:
	g.Expect(len(hydratedInfo.SecretIDs)).To(gomega.Equal(2))
}
