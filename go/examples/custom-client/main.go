// This example configures a client using a `CUSTOM_MODAL_ID` and `CUSTOM_MODAL_SECRET` environment variable.

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	ctx := context.Background()

	modalID := os.Getenv("CUSTOM_MODAL_ID")
	if modalID == "" {
		log.Fatal("CUSTOM_MODAL_ID environment variable not set")
	}
	modalSecret := os.Getenv("CUSTOM_MODAL_SECRET")
	if modalSecret == "" {
		log.Fatal("CUSTOM_MODAL_SECRET environment variable not set")
	}

	mc, err := modal.NewClientWithOptions(&modal.ClientParams{
		TokenID:     modalID,
		TokenSecret: modalSecret,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		log.Fatalf("Failed to get Function: %v", err)
	}
	fmt.Printf("%#v\n", echo)
}
