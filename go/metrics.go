package modal

import (
	"context"
	"strings"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// TimeSeriesPoint is an observation at a UTC-aligned bucket start.
type TimeSeriesPoint struct {
	Time time.Time
	// Value is nil for missing observations; observed zeroes remain zero.
	Value *float64
}

// MetricSeries contains an individual metric's ordered observations.
type MetricSeries struct {
	Name        string
	Unit        string
	Aggregation string
	Description string
	Points      []TimeSeriesPoint
}

// MetricDefinition describes an exportable series.
type MetricDefinition struct {
	Name        string
	Unit        string
	Description string
}

// MetricsSchema describes a group and all its individual series.
type MetricsSchema struct {
	Name        string
	Definitions []MetricDefinition
}

// ObjectMetrics contains samples over the effective UTC-aligned [Since, Until) range.
type ObjectMetrics struct {
	ObjectID   string
	Since      time.Time
	Until      time.Time
	BucketSize time.Duration
	// PointCount includes missing observations across all series.
	PointCount int
	Series     map[string]MetricSeries
}

// FunctionMetricsExportParams configures a metric export.
type FunctionMetricsExportParams struct {
	// Since defaults to one hour before Until.
	Since *time.Time
	// Until defaults to the current time.
	Until *time.Time
	// BucketSize is optional; nil lets Modal select a supported width.
	BucketSize *time.Duration
	// Groups selects metric groups; empty selects all groups.
	Groups []string
	// AllVariants includes the root Function and all its variants.
	AllVariants bool
}

// MetricsSchemaParams selects groups whose definitions should be returned.
type MetricsSchemaParams struct {
	// Groups defaults to all groups.
	Groups []string
}

// FunctionMetricsManager provides access to Function metrics.
type FunctionMetricsManager struct {
	client     *Client
	functionID string
}

func metricsWindow(sinceParam, untilParam *time.Time) (*timestamppb.Timestamp, *timestamppb.Timestamp, error) {
	until := time.Now().UTC()
	if untilParam != nil {
		until = untilParam.UTC()
	}
	since := until.Add(-time.Hour)
	if sinceParam != nil {
		since = sinceParam.UTC()
	}
	if !since.Before(until) {
		return nil, nil, InvalidError{Exception: "`since` must be before `until`."}
	}
	if since.Before(time.Unix(0, 0)) {
		return nil, nil, InvalidError{Exception: "`since` must be at or after the Unix epoch."}
	}
	if until.Sub(since) > 31*24*time.Hour {
		return nil, nil, InvalidError{Exception: "Metrics time range cannot exceed 31 days."}
	}
	return timestamppb.New(since), timestamppb.New(until), nil
}

func validateMetricsGroups(groups []string) error {
	for _, group := range groups {
		if group == "" {
			return InvalidError{Exception: "`groups` must contain nonempty group names."}
		}
	}
	return nil
}

func metricsRequest(params *FunctionMetricsExportParams) (*pb.FunctionGetMetricsRequest, error) {
	if params == nil {
		params = &FunctionMetricsExportParams{}
	}
	since, until, err := metricsWindow(params.Since, params.Until)
	if err != nil {
		return nil, err
	}
	if err := validateMetricsGroups(params.Groups); err != nil {
		return nil, err
	}
	request := pb.FunctionGetMetricsRequest_builder{Since: since, Until: until, Groups: params.Groups, Rollup: params.AllVariants}.Build()
	if params.BucketSize != nil {
		bucket := *params.BucketSize
		if bucket < time.Minute || bucket%time.Second != 0 || bucket/time.Second > 1<<32-1 {
			return nil, InvalidError{Exception: "`bucketSize` must be a whole number of seconds, at least one minute."}
		}
		request.SetBucketSecs(uint32(bucket / time.Second))
	}
	return request, nil
}

// Export fetches metric samples for a Function.
//
// Defaults to all groups over the last hour. Since defaults to one hour before
// Until. The maximum range is 31 days, with at most 500 points per series. If
// BucketSize is omitted, Modal selects a supported width targeting approximately
// 100 buckets. Bounds round down to UTC bucket boundaries, so data before Since
// may be included and the partial bucket at Until is excluded.
//
// Set AllVariants to true to include the root Function and its variants.
// Missing observations have a nil Value. Use [FunctionMetricsManager.Schema] to
// discover group names and series definitions without fetching samples.
//
//	bucketSize := 5 * time.Minute
//	metrics, err := function.Metrics.Export(ctx, &modal.FunctionMetricsExportParams{
//		BucketSize: &bucketSize,
//		Groups:     []string{"calls"},
//	})
//	if err != nil {
//		return err
//	}
//	for _, point := range metrics.Series["successful_calls_count"].Points {
//		if point.Value == nil {
//			fmt.Println(point.Time, "missing")
//		} else {
//			fmt.Println(point.Time, *point.Value)
//		}
//	}
func (m *FunctionMetricsManager) Export(ctx context.Context, params *FunctionMetricsExportParams) (*ObjectMetrics, error) {
	request, err := metricsRequest(params)
	if err != nil {
		return nil, err
	}
	request.SetFunctionId(m.functionID)
	response, err := m.client.cpClient.FunctionGetMetrics(ctx, request)
	if err != nil {
		return nil, err
	}
	result := &ObjectMetrics{ObjectID: m.functionID, Since: response.GetSince().AsTime(), Until: response.GetUntil().AsTime(), BucketSize: time.Duration(response.GetBucketSecs()) * time.Second, Series: make(map[string]MetricSeries)}
	for _, item := range response.GetSeries() {
		name := item.GetName()
		series := MetricSeries{Name: name, Unit: item.GetUnit(), Description: item.GetDescription(), Aggregation: name[strings.LastIndex(name, "_")+1:], Points: make([]TimeSeriesPoint, 0, len(item.GetPoints()))}
		for _, point := range item.GetPoints() {
			observation := TimeSeriesPoint{Time: point.GetTimestamp().AsTime()}
			if point.HasValue() {
				value := point.GetValue()
				observation.Value = &value
			}
			series.Points = append(series.Points, observation)
		}
		result.Series[name] = series
	}
	for _, series := range result.Series {
		result.PointCount += len(series.Points)
	}
	return result, nil
}

// Schema returns metric definitions keyed by group name without fetching samples.
// Pass nil params or omit Groups to return all supported groups. Each group's
// Definitions contains its individual series names, units, and descriptions.
// Use these group names in [FunctionMetricsExportParams.Groups] when exporting.
//
// Inspect the available call metrics:
//
//	schema, err := fn.Metrics.Schema(ctx, &modal.MetricsSchemaParams{
//		Groups: []string{"calls"},
//	})
//	if err != nil {
//		return err
//	}
//	for _, definition := range schema["calls"].Definitions {
//		fmt.Println(definition.Name, definition.Unit, definition.Description)
//	}
func (m *FunctionMetricsManager) Schema(ctx context.Context, params *MetricsSchemaParams) (map[string]MetricsSchema, error) {
	if params == nil {
		params = &MetricsSchemaParams{}
	}
	since, until, err := metricsWindow(nil, nil)
	if err != nil {
		return nil, err
	}
	if err := validateMetricsGroups(params.Groups); err != nil {
		return nil, err
	}
	response, err := m.client.cpClient.MetricsGetInfo(ctx, pb.MetricsGetInfoRequest_builder{
		TargetType: pb.MetricsGetInfoRequest_METRICS_TARGET_TYPE_FUNCTION,
		Since:      since, Until: until, Groups: params.Groups,
	}.Build())
	if err != nil {
		return nil, err
	}
	result := make(map[string]MetricsSchema)
	for _, row := range response.GetGroups() {
		group, ok := result[row.GetName()]
		if !ok {
			group = MetricsSchema{Name: row.GetName(), Definitions: []MetricDefinition{}}
		}
		for _, metric := range row.GetMetrics() {
			group.Definitions = append(group.Definitions, MetricDefinition{Name: metric.GetName(), Unit: metric.GetUnit(), Description: metric.GetDescription()})
		}
		result[row.GetName()] = group
	}
	return result, nil
}
