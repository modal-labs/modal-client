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
		Command: []string{"sh", "-c", "ls -la /mnt/s3-bucket"},
		CloudBucketMounts: map[string]*modal.CloudBucketMount{
			"/mnt/s3-bucket": cloudBucketMount,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	fmt.Printf("S3 Sandbox: %s\n", sb.SandboxID)

	output, err := io.ReadAll(sb.Stdout)
	if err != nil {
		log.Fatalf("Failed to read from Sandbox stdout: %v", err)
	}

	fmt.Printf("Sandbox directory listing of /mnt/s3-bucket:\n%s\n", string(output))
}
