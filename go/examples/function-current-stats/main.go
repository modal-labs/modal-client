// Demonstrates how to get current statistics for a Modal Function.

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

	function, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		log.Fatalf("Failed to get Function: %v", err)
	}

	stats, err := function.GetCurrentStats(ctx)
	if err != nil {
		log.Fatalf("Failed to get Function stats: %v", err)
	}

	fmt.Println("Function Statistics:")
	fmt.Printf("  Backlog: %d inputs\n", stats.Backlog)
	fmt.Printf("  Total Runners: %d containers\n", stats.NumTotalRunners)
}
