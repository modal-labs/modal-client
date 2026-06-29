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

	// Create a sandbox with only modal.com allowed (empty CIDR list blocks raw IP traffic).
	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:                 []string{"sleep", "infinity"},
		OutboundDomainAllowlist: []string{"modal.com"},
		OutboundCIDRAllowlist:   []string{},
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Println("Created Sandbox:", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox: %v", err)
		}
	}()

	// Try to reach example.com — should fail because only modal.com is allowed.
	p, err := sb.Exec(ctx, []string{"wget", "-q", "-O", "-", "--timeout=5", "http://example.com"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec: %v", err)
	}
	errOut, err := io.ReadAll(p.Stderr)
	if err != nil {
		log.Fatalf("Failed to read stderr: %v", err)
	}
	rc, err := p.Wait(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to wait: %v", err)
	}
	fmt.Printf("wget example.com (blocked): exit=%d stderr=%s\n", rc, string(errOut))

	// Unblock: widen the policy to allow all domains.
	err = sb.UpdateNetworkPolicy(ctx, &modal.SandboxUpdateNetworkPolicyParams{
		OutboundDomainAllowlist: &modal.Allowlist{Entries: []string{"*"}},
		OutboundCIDRAllowlist:   &modal.Allowlist{Entries: []string{"0.0.0.0/0"}},
	})
	if err != nil {
		log.Fatalf("Failed to update network policy: %v", err)
	}
	fmt.Println("Widened policy to allow all domains.")

	// Try again — should succeed now.
	p, err = sb.Exec(ctx, []string{"wget", "-q", "-O", "-", "--timeout=5", "http://example.com"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec: %v", err)
	}
	body, err := io.ReadAll(p.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	rc, err = p.Wait(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to wait: %v", err)
	}
	fmt.Printf("wget example.com (allowed): exit=%d body_len=%d\n", rc, len(body))
}
