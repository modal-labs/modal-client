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

	secret, err := mc.Secrets.FromMap(ctx, map[string]string{
		"CURL_VERSION": "8.12.1-r1",
	}, nil)
	if err != nil {
		log.Fatal(err)
	}

	image := mc.Images.FromRegistry("alpine:3.21", nil).
		DockerfileCommands([]string{"RUN apk add --no-cache curl=$CURL_VERSION"}, &modal.ImageDockerfileCommandsParams{
			Secrets: []*modal.Secret{secret},
		}).
		DockerfileCommands([]string{"ENV SERVER=ipconfig.me"}, nil)

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sh", "-c", "curl -Ls $SERVER"},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	fmt.Println("Created Sandbox with ID:", sb.SandboxID)

	output, err := io.ReadAll(sb.Stdout)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Sandbox output:", string(output))
}
