package test

import (
	"reflect"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

func TestFunctionCall(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, err := function.Remote(ctx, nil, map[string]any{"s": "hello"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))

	result, err = function.Remote(ctx, []any{"hello"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))
}

func TestFunctionCallPreCborVersionError(t *testing.T) {
	// test that calling a pre 1.2 function raises an error
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "test-support-1-1", "identity_with_repr", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = function.Remote(ctx, nil, map[string]any{"s": "hello"})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("Redeploy with Modal Python SDK >= 1.2"))
}

func TestFunctionCallGoMap(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "identity_with_repr", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	inputArg := map[string]any{"s": "hello"}
	result, err := function.Remote(ctx, []any{inputArg}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	resultSlice, ok := result.([]any)
	g.Expect(ok).Should(gomega.BeTrue())
	g.Expect(len(resultSlice)).Should(gomega.Equal(2))

	g.Expect(compareFlexible(resultSlice[0], inputArg)).Should(gomega.BeTrue())

	reprResult, ok := resultSlice[1].(string)
	g.Expect(ok).Should(gomega.BeTrue())
	g.Expect(reprResult).Should(gomega.Equal(`{'s': 'hello'}`))
}

func TestFunctionCallDateTimeRoundtrip(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "identity_with_repr", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	testTime := time.Date(2024, 1, 15, 10, 30, 45, 123456789, time.UTC)
	result, err := function.Remote(ctx, []any{testTime}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	// Parse the result - identity_with_repr returns [input, repr(input)]
	resultSlice, ok := result.([]any)
	g.Expect(ok).Should(gomega.BeTrue())
	g.Expect(len(resultSlice)).Should(gomega.Equal(2))

	reprResult, ok := resultSlice[1].(string)
	g.Expect(ok).Should(gomega.BeTrue())

	g.Expect(reprResult).Should(gomega.ContainSubstring("datetime.datetime"))
	g.Expect(reprResult).Should(gomega.ContainSubstring("2024"))

	receivedTime, ok := resultSlice[0].(time.Time)
	g.Expect(ok).Should(gomega.BeTrue())

	timeDiff := testTime.Sub(receivedTime)
	if timeDiff < 0 {
		timeDiff = -timeDiff
	}

	// Python's datetime has microsecond precision (not nanosecond)
	// CBOR encodes time.Time with TimeRFC3339Nano (nanosecond precision)
	// Python decodes to datetime (rounds to nearest microsecond)
	// When Python re-encodes, we get back microsecond precision
	// So we should expect to lose sub-microsecond precision (< 1000 ns)
	//
	// Our test uses 123456789 nanoseconds = 123.456789 milliseconds
	// Python will round to 123456 microseconds = 123.456 milliseconds
	// So we should lose exactly 789 nanoseconds
	g.Expect(timeDiff).Should(gomega.BeNumerically("<", time.Microsecond))

	g.Expect(receivedTime.Truncate(time.Microsecond)).Should(gomega.Equal(testTime.Truncate(time.Microsecond)))
}

func TestFunctionCallLargeInput(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "bytelength", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	len := 3 * 1000 * 1000 // More than 2 MiB, offload to blob storage
	input := make([]byte, len)
	result, err := function.Remote(ctx, []any{input}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal(uint64(len)))
}

func TestFunctionNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	_, err := tc.Functions.FromName(ctx, "libmodal-test-support", "not_a_real_function", nil)
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.NotFoundError{}))
}

func TestFunctionCallInputPlane(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "input_plane", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, err := function.Remote(ctx, []any{"hello"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))
}

func TestFunctionGetCurrentStats(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return pb.FunctionGetResponse_builder{
				FunctionId: "fid-stats",
			}.Build(), nil
		},
	)

	f, err := mock.Functions.FromName(ctx, "test-app", "test-function", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "/FunctionGetCurrentStats",
		func(req *pb.FunctionGetCurrentStatsRequest) (*pb.FunctionStats, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid-stats"))
			return pb.FunctionStats_builder{Backlog: 3, NumTotalTasks: 7}.Build(), nil
		},
	)

	stats, err := f.GetCurrentStats(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(stats).To(gomega.Equal(&modal.FunctionStats{Backlog: 3, NumTotalRunners: 7}))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestFunctionUpdateAutoscaler(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return pb.FunctionGetResponse_builder{
				FunctionId: "fid-auto",
			}.Build(), nil
		},
	)

	f, err := mock.Functions.FromName(ctx, "test-app", "test-function", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "/FunctionUpdateSchedulingParams",
		func(req *pb.FunctionUpdateSchedulingParamsRequest) (*pb.FunctionUpdateSchedulingParamsResponse, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid-auto"))
			s := req.GetSettings()
			g.Expect(s.GetMinContainers()).To(gomega.Equal(uint32(1)))
			g.Expect(s.GetMaxContainers()).To(gomega.Equal(uint32(10)))
			g.Expect(s.GetBufferContainers()).To(gomega.Equal(uint32(2)))
			g.Expect(s.GetScaledownWindow()).To(gomega.Equal(uint32(300)))
			return &pb.FunctionUpdateSchedulingParamsResponse{}, nil
		},
	)

	err = f.UpdateAutoscaler(ctx, &modal.FunctionUpdateAutoscalerParams{
		MinContainers:    ptrU32(1),
		MaxContainers:    ptrU32(10),
		BufferContainers: ptrU32(2),
		ScaledownWindow:  ptrU32(300),
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "/FunctionUpdateSchedulingParams",
		func(req *pb.FunctionUpdateSchedulingParamsRequest) (*pb.FunctionUpdateSchedulingParamsResponse, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid-auto"))
			g.Expect(req.GetSettings().GetMinContainers()).To(gomega.Equal(uint32(2)))
			return &pb.FunctionUpdateSchedulingParamsResponse{}, nil
		},
	)

	err = f.UpdateAutoscaler(ctx, &modal.FunctionUpdateAutoscalerParams{
		MinContainers: ptrU32(2),
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func ptrU32(v uint32) *uint32 { return &v }

func TestFunctionGetWebURL(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return pb.FunctionGetResponse_builder{
				FunctionId: "fid-normal",
			}.Build(), nil
		},
	)

	f, err := mock.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(f.GetWebURL()).To(gomega.Equal(""))

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			g.Expect(req.GetAppName()).To(gomega.Equal("libmodal-test-support"))
			g.Expect(req.GetObjectTag()).To(gomega.Equal("web_endpoint"))
			return pb.FunctionGetResponse_builder{
				FunctionId:     "fid-web",
				HandleMetadata: pb.FunctionHandleMetadata_builder{WebUrl: "https://endpoint.internal"}.Build(),
			}.Build(), nil
		},
	)

	wef, err := mock.Functions.FromName(ctx, "libmodal-test-support", "web_endpoint", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(wef.GetWebURL()).To(gomega.Equal("https://endpoint.internal"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestFunctionFromNameWithDotNotation(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	_, err := tc.Functions.FromName(ctx, "libmodal-test-support", "MyClass.myMethod", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.Equal("cannot retrieve Cls methods using Functions.FromName(). Use:\n  cls, _ := client.Cls.FromName(ctx, \"libmodal-test-support\", \"MyClass\", nil)\n  instance, _ := cls.Instance(ctx, nil)\n  m, _ := instance.Method(\"myMethod\")"))
}

func TestWebEndpointRemoteCallError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "web_endpoint_echo", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = function.Remote(ctx, []any{"hello"}, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
	g.Expect(err.Error()).Should(gomega.ContainSubstring("A webhook Function cannot be invoked for remote execution with 'Remote'"))
}

func TestWebEndpointSpawnCallError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(ctx, "libmodal-test-support", "web_endpoint_echo", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = function.Spawn(ctx, []any{"hello"}, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
	g.Expect(err.Error()).Should(gomega.ContainSubstring("A webhook Function cannot be invoked for remote execution with 'Spawn'"))
}

// compareFlexible compares two values with flexible type handling
func compareFlexible(a, b interface{}) bool {
	// Handle nil cases explicitly
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	// Handle map comparisons
	av := reflect.ValueOf(a)
	bv := reflect.ValueOf(b)

	if av.Kind() == reflect.Map && bv.Kind() == reflect.Map {
		return compareMaps(a, b)
	}
	// For other types, fall back to reflect.DeepEqual
	return reflect.DeepEqual(a, b)
}

// compareMaps compares two maps with flexible key type handling
func compareMaps(a, b interface{}) bool {
	av := reflect.ValueOf(a)
	bv := reflect.ValueOf(b)

	if av.Kind() != reflect.Map || bv.Kind() != reflect.Map {
		return false
	}

	if av.Len() != bv.Len() {
		return false
	}

	for _, key := range av.MapKeys() {
		aVal := av.MapIndex(key)

		// Try to find the corresponding key in b
		// Handle cases where key types might differ (string vs interface{})
		var bVal reflect.Value
		found := false

		for _, bKey := range bv.MapKeys() {
			if compareFlexible(key.Interface(), bKey.Interface()) {
				bVal = bv.MapIndex(bKey)
				found = true
				break
			}
		}

		if !found || !bVal.IsValid() {
			return false
		}

		// Use flexible comparison for values
		if !compareFlexible(aVal.Interface(), bVal.Interface()) {
			return false
		}
	}

	return true
}
