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

	volume, err := mc.Volumes.Ephemeral(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to create ephemeral Volume: %v", err)
	}
	defer volume.CloseEphemeral()

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sleep", "infinity"},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	writer, err := sb.ExperimentalSidecars.Create(ctx, "writer", image, &modal.SidecarCreateParams{
		Command: []string{"sh", "-c", "echo 'Hello from the writer sidecar!' > /mnt/volume/message.txt"},
		Volumes: map[string]*modal.Volume{"/mnt/volume": volume},
	})
	if err != nil {
		log.Fatalf("Failed to create writer sidecar: %v", err)
	}
	exitCode, err := writer.Wait(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to wait for writer sidecar: %v", err)
	}
	fmt.Printf("Writer sidecar finished with exit code: %d\n", exitCode)

	readOnly := true
	reader, err := sb.ExperimentalSidecars.Create(ctx, "reader", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
		Volumes: map[string]*modal.Volume{
			"/mnt/volume": volume.WithMountOptions(&modal.VolumeMountOptionsParams{ReadOnly: &readOnly}),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create reader sidecar: %v", err)
	}

	proc, err := reader.Exec(ctx, []string{"cat", "/mnt/volume/message.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec in reader sidecar: %v", err)
	}
	output, err := io.ReadAll(proc.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	if _, err := proc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for reader exec: %v", err)
	}
	fmt.Printf("Reader sidecar output: %s", string(output))
}
