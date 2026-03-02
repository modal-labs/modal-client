// This example spawns a Function defined in `libmodal_test_support.py`, and
// later gets its outputs.

package main

import (
	"context"
	"fmt"
	"log"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	ctx := context.Background()
	mc, err := modal.NewClient()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		log.Fatalf("Failed to get Function: %v", err)
	}

	fc, err := echo.Spawn(ctx, nil, map[string]any{"s": "Hello world!"})
	if err != nil {
		log.Fatalf("Failed to spawn Function: %v", err)
	}

	ret, err := fc.Get(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to get Function results: %v", err)
	}
	fmt.Println("Response:", ret)
}
