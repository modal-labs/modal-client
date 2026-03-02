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

	sandboxName := "libmodal-example-named-sandbox"

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Name:    sandboxName,
		Command: []string{"cat"},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	fmt.Printf("Created Sandbox with name: %s\n", sandboxName)
	fmt.Printf("Sandbox ID: %s\n", sb.SandboxID)

	_, err = mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Name:    sandboxName,
		Command: []string{"cat"},
	})
	if err != nil {
		if alreadyExistsErr, ok := err.(modal.AlreadyExistsError); ok {
			fmt.Printf("Trying to create one more Sandbox with the same name fails: %s\n", alreadyExistsErr.Exception)
		} else {
			log.Fatalf("Unexpected error: %v", err)
		}
	}

	sbFromName, err := mc.Sandboxes.FromName(ctx, "libmodal-example", sandboxName, nil)
	if err != nil {
		log.Fatalf("Failed to get Sandbox by name: %v", err)
	}
	fmt.Printf("Retrieved the same Sandbox from name: %s\n", sbFromName.SandboxID)

	_, err = sbFromName.Stdin.Write([]byte("hello, named Sandbox"))
	if err != nil {
		log.Fatalf("Failed to write to Sandbox stdin: %v", err)
	}
	err = sbFromName.Stdin.Close()
	if err != nil {
		log.Fatalf("Failed to close Sandbox stdin: %v", err)
	}

	fmt.Println("Reading output:")
	output, err := io.ReadAll(sbFromName.Stdout)
	if err != nil {
		log.Fatalf("Failed to read output from Sandbox stdout: %v", err)
	}
	fmt.Printf("%s\n", output)
}
