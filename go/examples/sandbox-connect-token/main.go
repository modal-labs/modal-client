package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"

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

	// Create a Sandbox with Python's built-in HTTP server
	image := mc.Images.FromRegistry("python:3.12-alpine", nil)

	// To use Sandbox Connect Tokens, the server must listen on port 8080.
	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"python3", "-m", "http.server", "8080"},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	creds, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{UserMetadata: "abc"})
	if err != nil {
		log.Fatalf("Failed to create connect token: %v", err)
	}
	fmt.Printf("Got url: %v, credentials: %v\n", creds.URL, creds.Token)

	fmt.Println("\nConnecting to HTTP server...")
	req, err := http.NewRequestWithContext(ctx, "GET", creds.URL, nil)
	if err != nil {
		log.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", creds.Token))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatalf("Failed to make request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	fmt.Printf("Response status: %d\n", resp.StatusCode)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Failed to read response body: %v", err)
	}

	fmt.Printf("Response body:\n%s\n", string(body))
}
