package modal

import (
	"context"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
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
