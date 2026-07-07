// This example calls a Modal Cls defined in `libmodal_test_support.py` and
// overrides its routing region, so that inputs/outputs are routed through a specific
// region at invocation time. Modal Functions can be similarly rerouted.

package main

import (
	"context"
	"fmt"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	ctx := context.Background()
	mc, _ := modal.NewClient()

	cls, _ := mc.Cls.FromName(ctx, "libmodal-test-support", "EchoClsInputPlane", nil)

	// Override the class's default routing region so that inputs/outputs are routed
	// through us-west.
	region := "us-west"
	instance, _ := cls.
		WithOptions(&modal.ClsWithOptionsParams{RoutingRegion: &region}).
		Instance(ctx, nil)

	method, _ := instance.Method("echo_string")

	result, _ := method.Remote(ctx, nil, map[string]any{"s": "hello"})
	fmt.Println(result)
}
