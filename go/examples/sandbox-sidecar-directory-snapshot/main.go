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

	image, err := mc.Images.FromRegistry("alpine:3.21", nil).Build(ctx, app, nil)
	if err != nil {
		log.Fatalf("Failed to build Image: %v", err)
	}

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sleep", "infinity"},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	container, err := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
	})
	if err != nil {
		log.Fatalf("Failed to create sidecar: %v", err)
	}
	if err := container.MountImage(ctx, "/workspace", nil, nil); err != nil {
		log.Fatalf("Failed to mount empty workspace: %v", err)
	}

	writeProc, err := container.Exec(ctx, []string{"sh", "-c", "echo persisted > /workspace/message.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to write workspace state: %v", err)
	}
	if _, err := writeProc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for workspace write: %v", err)
	}

	workspaceSnapshot, err := container.SnapshotDirectory(ctx, "/workspace", nil)
	if err != nil {
		log.Fatalf("Failed to snapshot workspace: %v", err)
	}

	if _, err := container.Terminate(ctx, &modal.SidecarTerminateParams{Wait: true}); err != nil {
		log.Fatalf("Failed to terminate sidecar: %v", err)
	}

	nextContainer, err := sb.ExperimentalSidecars.Create(ctx, "next-worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
	})
	if err != nil {
		log.Fatalf("Failed to create next sidecar: %v", err)
	}
	if err := nextContainer.MountImage(ctx, "/workspace", workspaceSnapshot, nil); err != nil {
		log.Fatalf("Failed to restore workspace: %v", err)
	}
	readProc, err := nextContainer.Exec(ctx, []string{"cat", "/workspace/message.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to read restored workspace: %v", err)
	}
	output, err := io.ReadAll(readProc.Stdout)
	if err != nil {
		log.Fatalf("Failed to read restored workspace output: %v", err)
	}
	if _, err := readProc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for workspace read: %v", err)
	}
	fmt.Printf("Restored workspace contains: %s", string(output))
	if err := nextContainer.UnmountImage(ctx, "/workspace", nil); err != nil {
		log.Fatalf("Failed to unmount restored workspace: %v", err)
	}
	if _, err := nextContainer.Terminate(ctx, &modal.SidecarTerminateParams{Wait: true}); err != nil {
		log.Fatalf("Failed to terminate next sidecar: %v", err)
	}
}
