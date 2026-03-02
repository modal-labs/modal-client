// This example calls a function defined in `libmodal_test_support.py`.

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

	ret, err := echo.Remote(ctx, []any{"Hello world!"}, nil)
	if err != nil {
		log.Fatalf("Failed to call Function: %v", err)
	}
	fmt.Println("Response:", ret)

	ret, err = echo.Remote(ctx, nil, map[string]any{"s": "Hello world!"})
	if err != nil {
		log.Fatalf("Failed to call Function with kwargs: %v", err)
	}
	fmt.Println("Response:", ret)
}
