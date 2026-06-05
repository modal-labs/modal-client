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
	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)
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
	fmt.Printf("Started sidecar: %s\n", container.ContainerID)

	proc, err := container.Exec(
		ctx,
		[]string{"sh", "-c", `echo "$GREETING from sidecar"`},
		&modal.SidecarExecParams{
			Env: map[string]string{"GREETING": "hello"},
		},
	)
	if err != nil {
		log.Fatalf("Failed to exec in sidecar: %v", err)
	}
	output, err := io.ReadAll(proc.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	if _, err := proc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for sidecar exec: %v", err)
	}
	fmt.Printf("Sidecar said: %s", string(output))

	exitCode, err := container.Terminate(ctx, &modal.SidecarTerminateParams{Wait: true})
	if err != nil {
		log.Fatalf("Failed to terminate sidecar: %v", err)
	}
	fmt.Printf("Sidecar terminated with exit code: %d\n", exitCode)
}
