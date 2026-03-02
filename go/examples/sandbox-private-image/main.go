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

	secret, err := mc.Secrets.FromName(ctx, "libmodal-aws-ecr-test", &modal.SecretFromNameParams{
		RequiredKeys: []string{"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"},
	})
	if err != nil {
		log.Fatalf("Failed to get Secret: %v", err)
	}

	image := mc.Images.FromAwsEcr("459781239556.dkr.ecr.us-east-1.amazonaws.com/ecr-private-registry-test-7522615:python", secret)

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"python", "-c", `import sys; sys.stdout.write(sys.stdin.read())`},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Sandbox: %s\n", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	_, err = sb.Stdin.Write([]byte("this is input that should be mirrored by the Python one-liner"))
	if err != nil {
		log.Fatalf("Failed to write to Sandbox stdin: %v", err)
	}
	err = sb.Stdin.Close()
	if err != nil {
		log.Fatalf("Failed to close Sandbox stdin: %v", err)
	}

	output, err := io.ReadAll(sb.Stdout)
	if err != nil {
		log.Fatalf("Failed to read from Sandbox stdout: %v", err)
	}

	fmt.Printf("output: %s\n", string(output))
}
