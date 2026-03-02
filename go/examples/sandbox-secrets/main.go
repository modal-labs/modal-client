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

	secret, err := mc.Secrets.FromName(ctx, "libmodal-test-secret", &modal.SecretFromNameParams{RequiredKeys: []string{"c"}})
	if err != nil {
		log.Fatalf("Failed finding a Secret: %v", err)
	}

	ephemeralSecret, err := mc.Secrets.FromMap(ctx, map[string]string{
		"d": "123",
	}, nil)
	if err != nil {
		log.Fatalf("Failed creating ephemeral Secret: %v", err)
	}

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sh", "-lc", "printenv | grep -E '^c|d='"}, Secrets: []*modal.Secret{secret, ephemeralSecret},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Sandbox created: %s\n", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	output, err := io.ReadAll(sb.Stdout)
	if err != nil {
		log.Fatalf("Failed to read output: %v", err)
	}
	fmt.Printf("Sandbox environment variables from Secrets:\n%v\n", string(output))
}
