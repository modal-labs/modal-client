package main

import (
	"context"
	"fmt"
	"io"
	"log"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	ctx := context.Background()
	mc, err := modal.NewClient()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	app, err := mc.Apps.FromName(ctx, "libmodal-example", &modal.AppFromNameParams{CreateIfMissing: true})
	if err != nil {
		log.Fatalf("Failed to get or create App: %v", err)
	}

	image := mc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"cat"},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Created Sandbox with ID: %s\n", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	sbFromID, err := mc.Sandboxes.FromID(ctx, sb.SandboxID)
	if err != nil {
		log.Fatalf("Failed to get Sandbox with ID: %v", err)
	}
	fmt.Printf("Queried Sandbox with ID: %v\n", sbFromID.SandboxID)

	_, err = sb.Stdin.Write([]byte("this is input that should be mirrored by cat"))
	if err != nil {
		log.Fatalf("Failed to write to Sandbox stdin: %v", err)
	}
	err = sb.Stdin.Close()
	if err != nil {
		log.Fatalf("Failed to close Sandbox stdin: %v", err)
	}

	output, err := io.ReadAll(sb.Stdout)
	if err != nil {
		log.Fatalf("Failed to read from Sandbox stdout: %v", err)
	}

	fmt.Printf("output: %s\n", string(output))
}
