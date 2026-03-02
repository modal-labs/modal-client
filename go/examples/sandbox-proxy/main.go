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

	image := mc.Images.FromRegistry("alpine/curl:8.14.1", nil)

	proxy, err := mc.Proxies.FromName(ctx, "libmodal-test-proxy", &modal.ProxyFromNameParams{Environment: "libmodal"})
	if err != nil {
		log.Fatalf("Failed to get Proxy: %v", err)
	}
	fmt.Printf("Using Proxy: %s\n", proxy.ProxyID)

	sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Proxy: proxy,
	})
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	fmt.Printf("Created Sandbox with proxy: %s\n", sb.SandboxID)
	defer func() {
		if _, err := sb.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	p, err := sb.Exec(ctx, []string{"curl", "-s", "ifconfig.me"}, nil)
	if err != nil {
		log.Fatalf("Failed to start IP fetch command: %v", err)
	}

	ip, err := io.ReadAll(p.Stdout)
	if err != nil {
		log.Fatalf("Failed to read IP output: %v", err)
	}

	fmt.Printf("External IP: %s\n", string(ip))
}
