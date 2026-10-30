package modal

import (
	"context"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type mockMetricsClient struct {
	pb.ModalClientClient
	export func(context.Context, *pb.FunctionGetMetricsRequest) (*pb.FunctionGetMetricsResponse, error)
	schema func(context.Context, *pb.MetricsGetInfoRequest) (*pb.MetricsGetInfoResponse, error)
}

func (m *mockMetricsClient) FunctionGetMetrics(ctx context.Context, req *pb.FunctionGetMetricsRequest, _ ...grpc.CallOption) (*pb.FunctionGetMetricsResponse, error) {
	return m.export(ctx, req)
}
func (m *mockMetricsClient) MetricsGetInfo(ctx context.Context, req *pb.MetricsGetInfoRequest, _ ...grpc.CallOption) (*pb.MetricsGetInfoResponse, error) {
	return m.schema(ctx, req)
}

func TestMetricsExport(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	start := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	end := start.Add(time.Hour)
	since, until := start.Add(30*time.Second), end.Add(30*time.Second)
	bucket := time.Minute
	zero := 0.0
	mock := &mockMetricsClient{export: func(_ context.Context, req *pb.FunctionGetMetricsRequest) (*pb.FunctionGetMetricsResponse, error) {
		g.Expect(req.GetFunctionId()).To(Equal("fu-test"))
		g.Expect(req.GetSince().AsTime()).To(Equal(since))
		g.Expect(req.GetUntil().AsTime()).To(Equal(until))
		g.Expect(req.GetBucketSecs()).To(Equal(uint32(60)))
		g.Expect(req.GetRollup()).To(BeTrue())
		g.Expect(req.GetGroups()).To(Equal([]string{"calls"}))
		return pb.FunctionGetMetricsResponse_builder{Since: timestamppb.New(start), Until: timestamppb.New(end), BucketSecs: 60, Series: []*pb.FunctionMetricSeries{
			pb.FunctionMetricSeries_builder{Name: "successful_calls_count", Unit: "calls", Description: "Successful inputs.", Points: []*pb.FunctionMetricPoint{
				pb.FunctionMetricPoint_builder{Timestamp: timestamppb.New(start), Value: &zero}.Build(),
				pb.FunctionMetricPoint_builder{Timestamp: timestamppb.New(start.Add(time.Minute))}.Build(),
			}}.Build(),
		}}.Build(), nil
	}}
	fn := newFunction(&Client{cpClient: &clientWithConn{ModalClientClient: mock}}, "fu-test", nil, nil, nil)
	result, err := fn.Metrics.Export(context.Background(), &FunctionMetricsExportParams{Since: &since, Until: &until, BucketSize: &bucket, Groups: []string{"calls"}, AllVariants: true})
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(result.ObjectID).To(Equal("fu-test"))
	g.Expect(result.Since).To(Equal(start))
	g.Expect(result.Until).To(Equal(end))
	g.Expect(result.BucketSize).To(Equal(time.Minute))
	g.Expect(result.PointCount).To(Equal(2))
	g.Expect(result.Series["successful_calls_count"]).To(Equal(MetricSeries{Name: "successful_calls_count", Unit: "calls", Aggregation: "count", Description: "Successful inputs.", Points: []TimeSeriesPoint{{Time: start, Value: &zero}, {Time: start.Add(time.Minute)}}}))
}

func TestMetricsExportDefaults(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	before := time.Now().Add(-time.Second)
	mock := &mockMetricsClient{export: func(_ context.Context, req *pb.FunctionGetMetricsRequest) (*pb.FunctionGetMetricsResponse, error) {
		g.Expect(req.GetUntil().AsTime().Sub(req.GetSince().AsTime())).To(Equal(time.Hour))
		g.Expect(req.GetUntil().AsTime().Before(before)).To(BeFalse())
		g.Expect(req.HasBucketSecs()).To(BeFalse())
		g.Expect(req.GetRollup()).To(BeFalse())
		g.Expect(req.GetGroups()).To(BeEmpty())
		return pb.FunctionGetMetricsResponse_builder{Since: req.GetSince(), Until: req.GetUntil(), BucketSecs: 60}.Build(), nil
	}}
	result, err := newFunction(&Client{cpClient: &clientWithConn{ModalClientClient: mock}}, "fu-test", nil, nil, nil).Metrics.Export(context.Background(), nil)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(result.Series).To(BeEmpty())
	g.Expect(result.PointCount).To(BeZero())
}

func TestMetricsSchema(t *testing.T) {
	t.Parallel()
	for _, groups := range [][]string{nil, {"calls"}} {
		t.Run("selection", func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			mock := &mockMetricsClient{schema: func(_ context.Context, req *pb.MetricsGetInfoRequest) (*pb.MetricsGetInfoResponse, error) {
				g.Expect(req.GetTargetType()).To(Equal(pb.MetricsGetInfoRequest_METRICS_TARGET_TYPE_FUNCTION))
				g.Expect(req.GetGroups()).To(Equal(groups))
				return pb.MetricsGetInfoResponse_builder{Groups: []*pb.MetricGroupDefinition{
					pb.MetricGroupDefinition_builder{Name: "calls", Metrics: []*pb.MetricDefinition{pb.MetricDefinition_builder{Name: "a_avg", Unit: "seconds", Description: "A"}.Build()}}.Build(),
					pb.MetricGroupDefinition_builder{Name: "calls", Metrics: []*pb.MetricDefinition{pb.MetricDefinition_builder{Name: "b_count", Unit: "calls", Description: "B"}.Build()}}.Build(),
				}}.Build(), nil
			}}
			result, err := newFunction(&Client{cpClient: &clientWithConn{ModalClientClient: mock}}, "fu-test", nil, nil, nil).Metrics.Schema(context.Background(), &MetricsSchemaParams{Groups: groups})
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(result).To(Equal(map[string]MetricsSchema{"calls": {Name: "calls", Definitions: []MetricDefinition{{Name: "a_avg", Unit: "seconds", Description: "A"}, {Name: "b_count", Unit: "calls", Description: "B"}}}}))
		})
	}
}

func TestMetricsValidation(t *testing.T) {
	t.Parallel()
	start := time.Unix(1800000000, 0)
	tooLate := start.Add(31*24*time.Hour + time.Nanosecond)
	invalidBucket := 60*time.Second + time.Nanosecond
	cases := []struct {
		name   string
		params *FunctionMetricsExportParams
	}{
		{"order", &FunctionMetricsExportParams{Since: &start, Until: &start}},
		{"range", &FunctionMetricsExportParams{Since: &start, Until: &tooLate}},
		{"bucket", &FunctionMetricsExportParams{BucketSize: &invalidBucket}},
		{"group", &FunctionMetricsExportParams{Groups: []string{""}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			_, err := newFunction(&Client{cpClient: &clientWithConn{ModalClientClient: &mockMetricsClient{}}}, "fu-test", nil, nil, nil).Metrics.Export(context.Background(), tc.params)
			g.Expect(err).To(BeAssignableToTypeOf(InvalidError{}))
		})
	}
}

func TestMetricsPropagateErrors(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	expected := status.Error(codes.InvalidArgument, "Invalid metrics groups")
	mock := &mockMetricsClient{
		export: func(ctx context.Context, _ *pb.FunctionGetMetricsRequest) (*pb.FunctionGetMetricsResponse, error) {
			g.Expect(ctx.Err()).To(MatchError(context.Canceled))
			return nil, expected
		},
		schema: func(ctx context.Context, _ *pb.MetricsGetInfoRequest) (*pb.MetricsGetInfoResponse, error) {
			g.Expect(ctx.Err()).To(MatchError(context.Canceled))
			return nil, expected
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	manager := newFunction(&Client{cpClient: &clientWithConn{ModalClientClient: mock}}, "fu-test", nil, nil, nil).Metrics
	_, err := manager.Export(ctx, nil)
	g.Expect(err).To(MatchError(expected))
	_, err = manager.Schema(ctx, nil)
	g.Expect(err).To(MatchError(expected))
}

func TestMetricsExportExplicitUntilAndServerLimits(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	until := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	mock := &mockMetricsClient{export: func(_ context.Context, req *pb.FunctionGetMetricsRequest) (*pb.FunctionGetMetricsResponse, error) {
		g.Expect(req.GetUntil().AsTime()).To(Equal(until))
		g.Expect(req.GetSince().AsTime()).To(Equal(until.Add(-time.Hour)))
		return pb.FunctionGetMetricsResponse_builder{Since: req.GetSince(), Until: req.GetUntil(), BucketSecs: 60}.Build(), nil
	}}
	fn := newFunction(&Client{cpClient: &clientWithConn{ModalClientClient: mock}}, "fu-test", nil, nil, nil)
	_, err := fn.Metrics.Export(context.Background(), &FunctionMetricsExportParams{Until: &until})
	g.Expect(err).NotTo(HaveOccurred())
	since := until.Add(-31 * 24 * time.Hour)
	bucket := time.Minute
	expected := status.Error(codes.InvalidArgument, "Metrics are limited to 500 points per series")
	mock.export = func(_ context.Context, req *pb.FunctionGetMetricsRequest) (*pb.FunctionGetMetricsResponse, error) {
		g.Expect(req.GetSince().AsTime()).To(Equal(since))
		g.Expect(req.GetBucketSecs()).To(Equal(uint32(60)))
		return nil, expected
	}
	_, err = fn.Metrics.Export(context.Background(), &FunctionMetricsExportParams{Since: &since, Until: &until, BucketSize: &bucket})
	g.Expect(err).To(MatchError(expected))
}
