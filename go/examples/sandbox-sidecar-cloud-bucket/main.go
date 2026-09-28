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

	secret, err := mc.Secrets.FromName(ctx, "libmodal-aws-bucket-secret", nil)
	if err != nil {
		log.Fatalf("Failed to get Secret: %v", err)
	}

	keyPrefix := "data/"
	cloudBucketMount, err := mc.CloudBucketMounts.New("my-s3-bucket", &modal.CloudBucketMountParams{
		Secret:    secret,
		KeyPrefix: &keyPrefix,
		ReadOnly:  true,
	})
	if err != nil {
		log.Fatalf("Failed to create Cloud Bucket Mount: %v", err)
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

	sidecar, err := sb.ExperimentalSidecars.Create(ctx, "reader", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
		CloudBucketMounts: map[string]*modal.CloudBucketMount{
			"/mnt/s3-bucket": cloudBucketMount,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create sidecar: %v", err)
	}

	proc, err := sidecar.Exec(ctx, []string{"ls", "-la", "/mnt/s3-bucket"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec in sidecar: %v", err)
	}
	output, err := io.ReadAll(proc.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	if _, err := proc.Wait(ctx, nil); err != nil {
		log.Fatalf("Failed to wait for exec: %v", err)
	}
	fmt.Printf("Sidecar directory listing of /mnt/s3-bucket:\n%s\n", string(output))
}
