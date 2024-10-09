package main

import (
	"context"
	"fmt"
	"log"

	modal "github.com/modal-labs/modal-client/pkg/modal"
)

func main() {
	// Load configuration
	config, err := modal.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Connect to the client
	client, err := modal.Connect(*config, "")
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}

	// Lookup the function
	ctx := context.Background()
	function, err := client.LookupFunction(ctx, "main", "payload-value", "f")
	if err != nil {
		log.Fatalf("Failed to lookup function: %v", err)
	}

	// Prepare arguments
	input := "hello-from-go"
	fmt.Printf("input = %v\n", input)
	args := modal.NewArgs(input)

	// Call the function
	result, err := function.Call(ctx, args)
	if err != nil {
		log.Fatalf("Function call failed: %v", err)
	}

	// Print the result
	fmt.Printf("result = %+v\n", *result.String)
}
