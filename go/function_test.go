package modal

import (
	"context"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

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
