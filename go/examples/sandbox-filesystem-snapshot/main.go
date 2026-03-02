package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

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

	baseImage := mc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := mc.Sandboxes.Create(ctx, app, baseImage, &modal.SandboxCreateParams{})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)

	sbFromID, err := mc.Sandboxes.FromID(ctx, sb.SandboxID)
	if err != nil {
		log.Fatalf("Failed to get Sandbox: %v", err)
	}
	defer func() {
		if _, err := sbFromID.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	_, err = sb.Exec(ctx, []string{"mkdir", "-p", "/app/data"}, nil)
	if err != nil {
		log.Fatalf("Failed to create directory: %v", err)
	}

	_, err = sb.Exec(ctx, []string{"sh", "-c", "echo 'This file was created in the first Sandbox' > /app/data/info.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	fmt.Println("Created file in first Sandbox")

	snapshotImage, err := sb.SnapshotFilesystem(ctx, 55*time.Second)
	if err != nil {
		log.Fatalf("Failed to snapshot filesystem: %v", err)
	}
	fmt.Printf("Filesystem snapshot created with Image ID: %s\n", snapshotImage.ImageID)

	_, err = sb.Terminate(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
	}

	// Create new Sandbox from snapshot Image
	sb2, err := mc.Sandboxes.Create(ctx, app, snapshotImage, nil)
	if err != nil {
		log.Fatalf("Failed to create Sandbox from snapshot: %v", err)
	}
	fmt.Printf("Started new Sandbox from snapshot: %s\n", sb2.SandboxID)

	defer func() {
		if _, err := sb2.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb2.SandboxID, err)
		}
	}()

	proc, err := sb2.Exec(ctx, []string{"cat", "/app/data/info.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec cat command: %v", err)
	}

	content, err := io.ReadAll(proc.Stdout)
	if err != nil {
		log.Fatalf("Failed to read output: %v", err)
	}
	fmt.Printf("File data read in second Sandbox: %s\n", string(content))
}
