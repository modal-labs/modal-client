package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

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

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:        []string{"python3", "-m", "http.server", "8000"},
		EncryptedPorts: []int{8000},
		Timeout:        1 * time.Minute,
		IdleTimeout:    30 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	fmt.Printf("Sandbox created: %s\n", sb.SandboxID)

	fmt.Println("Waiting for server to start...")
	time.Sleep(3 * time.Second)

	fmt.Println("Getting tunnel information...")
	tunnels, err := sb.Tunnels(ctx, 30*time.Second)
	if err != nil {
		log.Fatalf("Failed to get tunnels: %v", err)
	}

	tunnel := tunnels[8000]
	if tunnel == nil {
		log.Fatalf("No tunnel found for port 8000")
	}

	fmt.Println("Tunnel information:")
	fmt.Printf("  URL: %s\n", tunnel.URL())
	fmt.Printf("  Port: %d\n", tunnel.Port)

	fmt.Printf("\nMaking GET request to the tunneled server at %s\n", tunnel.URL())

	// Make a GET request to the tunneled server
	resp, err := http.Get(tunnel.URL())
	if err != nil {
		log.Fatalf("Failed to make GET request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		log.Fatalf("HTTP error! status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Failed to read response body: %v", err)
	}

	// Display first 500 characters of the response
	bodyStr := string(body)
	if len(bodyStr) > 500 {
		bodyStr = bodyStr[:500]
	}

	fmt.Printf("\nDirectory listing from server (first 500 chars):\n%s\n", bodyStr)

	fmt.Println("\n✅ Successfully connected to the tunneled server!")
}
