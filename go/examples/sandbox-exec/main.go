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

	image := mc.Images.FromRegistry("python:3.13-slim", nil)

	sb, err := mc.Sandboxes.Create(ctx, app, image, nil)
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Println("Started Sandbox:", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	p, err := sb.Exec(ctx,
		[]string{
			"python",
			"-c",
			`
import time
import sys
for i in range(50000):
	if i % 1000 == 0:
		time.sleep(0.01)
	print(i)
	print(i, file=sys.stderr)`,
		},
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to execute command in Sandbox: %v", err)
	}

	contentStdout, err := io.ReadAll(p.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	contentStderr, err := io.ReadAll(p.Stderr)
	if err != nil {
		log.Fatalf("Failed to read stderr: %v", err)
	}

	fmt.Printf("Got %d bytes stdout and %d bytes stderr\n", len(contentStdout), len(contentStderr))
	returnCode, err := p.Wait(ctx)
	if err != nil {
		log.Fatalf("Failed to wait for process completion: %v", err)
	}
	fmt.Println("Return code:", returnCode)

	secret, err := mc.Secrets.FromName(ctx, "libmodal-test-secret", &modal.SecretFromNameParams{RequiredKeys: []string{"c"}})
	if err != nil {
		log.Fatalf("Unable to get Secret: %v", err)
	}

	// Passing Secrets in a command
	p, err = sb.Exec(ctx, []string{"printenv", "c"}, &modal.SandboxExecParams{Secrets: []*modal.Secret{secret}})
	if err != nil {
		log.Fatalf("Faield to execute env command in Sandbox: %v", err)
	}

	secretStdout, err := io.ReadAll(p.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	fmt.Printf("Got environment variable c=%v\n", string(secretStdout))
}
