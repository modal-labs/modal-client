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
