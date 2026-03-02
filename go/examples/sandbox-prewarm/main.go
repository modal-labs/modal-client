// We use `Image.Build` to create an Image object on Modal
// that eagerly pulls from the registry. The first Sandbox created with this Image
// will ues this "pre-warmed" Image and will start faster.
package main

import (
	"context"
	"fmt"
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

	// With `.Build(app)`, we create an Image object on Modal that eagerly pulls
	// from the registry.
	image, err := mc.Images.FromRegistry("alpine:3.21", nil).Build(ctx, app)
	if err != nil {
		log.Fatalf("Unable to build Image: %v", err)
	}
	fmt.Printf("Image has ID: %v\n", image.ImageID)

	// You can save the ImageId and create a new Image object that referes to it.
	imageID := image.ImageID
	image2, err := mc.Images.FromID(ctx, imageID)
	if err != nil {
		log.Fatalf("Unable to look up Image from ID: %v", err)
	}

	sb, err := mc.Sandboxes.Create(ctx, app, image2, &modal.SandboxCreateParams{
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
	fmt.Printf("Sandbox: %s\n", sb.SandboxID)
}
