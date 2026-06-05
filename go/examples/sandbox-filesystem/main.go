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

	app, err := mc.Apps.FromName(ctx, "libmodal-example", &modal.AppFromNameParams{CreateIfMissing: true})
	if err != nil {
		log.Fatalf("Failed to get or create App: %v", err)
	}

	image := mc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)

	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	// Write a file
	if err := sb.Filesystem.WriteText(ctx, "Hello, Modal filesystem!\n", "/tmp/example.txt", nil); err != nil {
		log.Fatalf("Failed to write file: %v", err)
	}

	// Read the file
	content, err := sb.Filesystem.ReadText(ctx, "/tmp/example.txt", nil)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}
	fmt.Printf("File content:\n%s\n", content)
}
