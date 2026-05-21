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

	volume, err := mc.Volumes.FromName(ctx, "libmodal-example-volume", &modal.VolumeFromNameParams{
		CreateIfMissing: true,
	})
	if err != nil {
		log.Fatalf("Failed to create Volume: %v", err)
	}

	writerSandbox, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{
			"sh",
			"-c",
			"mkdir -p /mnt/volume/data && echo 'Hello from writer Sandbox!' > /mnt/volume/data/message.txt",
		},
		Volumes: map[string]*modal.Volume{
			"/mnt/volume": volume,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create writer Sandbox: %v", err)
	}
	fmt.Printf("Writer Sandbox: %s\n", writerSandbox.SandboxID)
	defer func() {
		if _, err := writerSandbox.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", writerSandbox.SandboxID, err)
		}
	}()

	exitCode, err := writerSandbox.Wait(ctx)
	if err != nil {
		log.Fatalf("Failed to wait for writer Sandbox: %v", err)
	}
	fmt.Printf("Writer finished with exit code: %d\n", exitCode)

	// Mount the Volume read-only and scoped to the /data subdirectory, so the
	// reader sees the file directly at /mnt/volume/message.txt.
	readOnly := true
	subPath := "/data"
	readerSandbox, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Volumes: map[string]*modal.Volume{
			"/mnt/volume": volume.WithMountOptions(&modal.VolumeMountOptions{
				ReadOnly: &readOnly,
				SubPath:  &subPath,
			}),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create reader Sandbox: %v", err)
	}
	fmt.Printf("Reader Sandbox: %s\n", readerSandbox.SandboxID)
	defer func() {
		if _, err := readerSandbox.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", readerSandbox.SandboxID, err)
		}
	}()

	rp, err := readerSandbox.Exec(ctx, []string{"cat", "/mnt/volume/message.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec read command: %v", err)
	}
	readOutput, err := io.ReadAll(rp.Stdout)
	if err != nil {
		log.Fatalf("Failed to read output: %v", err)
	}
	fmt.Printf("Reader output: %s", string(readOutput))

	wp, err := readerSandbox.Exec(ctx, []string{"sh", "-c", "echo 'This should fail' >> /mnt/volume/message.txt"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec write command: %v", err)
	}

	writeExitCode, err := wp.Wait(ctx)
	if err != nil {
		log.Fatalf("Failed to wait for write process: %v", err)
	}
	writeStderr, err := io.ReadAll(wp.Stderr)
	if err != nil {
		log.Fatalf("Failed to read stderr: %v", err)
	}

	fmt.Printf("Write attempt exit code: %d\n", writeExitCode)
	fmt.Printf("Write attempt stderr: %s", string(writeStderr))
}
