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

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)

	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	// Write a file
	writeFile, err := sb.Open(ctx, "/tmp/example.txt", "w")
	if err != nil {
		log.Fatalf("Failed to open file for writing: %v", err)
	}

	_, err = writeFile.Write([]byte("Hello, Modal filesystem!\n"))
	if err != nil {
		log.Fatalf("Failed to write to file: %v", err)
	}

	if err := writeFile.Close(); err != nil {
		log.Fatalf("Failed to close file: %v", err)
	}

	// Read the file
	reader, err := sb.Open(ctx, "/tmp/example.txt", "r")
	if err != nil {
		log.Fatalf("Failed to open file for reading: %v", err)
	}

	content, err := io.ReadAll(reader)
	if err != nil && err != io.EOF {
		log.Fatalf("Failed to read file: %v", err)
	}

	fmt.Printf("File content:\n%s\n", string(content))
	if err := reader.Close(); err != nil {
		log.Fatalf("Failed to close file: %v", err)
	}
}
