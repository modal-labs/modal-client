// This example demonstrates how to add custom telemetry and tracing to Modal API calls
// using gRPC interceptors. It shows a simple custom interceptor, but the same principle
// could also be used with e.g. OpenTelemetry for distributed tracing.

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"google.golang.org/grpc"
)

// telemetryInterceptor is a custom interceptor that measures API call latency
// and logs method names with timing information.
func telemetryInterceptor(
	ctx context.Context,
	method string,
	req, reply any,
	cc *grpc.ClientConn,
	invoker grpc.UnaryInvoker,
	opts ...grpc.CallOption,
) error {
	start := time.Now()
	err := invoker(ctx, method, req, reply, cc, opts...)
	duration := time.Since(start)

	status := "success"
	if err != nil {
		status = "error"
	}
	fmt.Printf("[TELEMETRY] method=%s duration=%v status=%s\n", method, duration, status)
	// You could also send this data to your observability backend, etc.

	return err
}

func main() {
	ctx := context.Background()

	fmt.Println("Initializing Modal client with telemetry interceptor. All API calls will be logged with timing information.")

	mc, err := modal.NewClientWithOptions(&modal.ClientParams{
		GRPCUnaryInterceptors: []grpc.UnaryClientInterceptor{
			telemetryInterceptor,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer mc.Close()

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		log.Fatalf("Failed to get Function: %v", err)
	}

	result, err := echo.Remote(ctx, nil, map[string]any{"s": "Hello from telemetry example!"})
	if err != nil {
		log.Fatalf("Failed to call function: %v", err)
	}
	fmt.Printf("Result: %v\n", result)

	fmt.Println("\nAll operations completed. See the telemetry logs above!")
}
