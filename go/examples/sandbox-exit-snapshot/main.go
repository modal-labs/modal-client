// This example creates a Sandbox with exit snapshots enabled, terminates it,
// and then starts a new Sandbox from the filesystem captured at exit.

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

	baseImage := mc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := mc.Sandboxes.Create(ctx, app, baseImage, &modal.SandboxCreateParams{
		ExperimentalOptions: map[string]any{"enable_exit_snapshot": true},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)

	mkdirProc, err := sb.Exec(ctx, []string{"mkdir", "-p", "/app/data"}, nil)
	if err != nil {
		log.Fatalf("Failed to create directory: %v", err)
	}
	if _, err = mkdirProc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for mkdir: %v", err)
	}

	echoProc, err := sb.Exec(ctx, []string{"sh", "-c", "echo 'This file was created in the first Sandbox' > /app/data/info.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	if _, err = echoProc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for echo: %v", err)
	}
	fmt.Println("Created file in first Sandbox")

	_, err = sb.Terminate(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
	}
	fmt.Println("Terminated first Sandbox")

	exitSnapshotImage, err := sb.ExperimentalGetExitSnapshot(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to get exit snapshot: %v", err)
	}
	fmt.Printf("Exit snapshot created with Image ID: %s\n", exitSnapshotImage.ImageID)

	sb2, err := mc.Sandboxes.Create(ctx, app, exitSnapshotImage, nil)
	if err != nil {
		log.Fatalf("Failed to create Sandbox from exit snapshot: %v", err)
	}
	fmt.Printf("Started new Sandbox from exit snapshot: %s\n", sb2.SandboxID)

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
