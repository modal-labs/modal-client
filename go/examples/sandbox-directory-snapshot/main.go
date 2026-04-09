// This example demonstrates the directory snapshots feature, which allows you to:
// - Take a snapshot of a directory in a Sandbox using `Sandbox.SnapshotDirectory`,
//   which will create a new Modal Image.
// - Mount a Modal Image at a specific directory within an already running Sandbox
//   using `Sandbox.MountImage`.
//
// For example, you can use this to mount user specific dependencies into a running
// Sandbox, that is started with a base Image with shared system dependencies. This
// way, you can update system dependencies and user projects independently.

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

	baseImage := mc.Images.FromRegistry("alpine:3.21", nil).DockerfileCommands([]string{
		"RUN apk add --no-cache git",
	}, nil)

	sb, err := mc.Sandboxes.Create(ctx, app, baseImage, nil)
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	sbFromID, err := mc.Sandboxes.FromID(ctx, sb.SandboxID)
	if err != nil {
		log.Fatalf("Failed to get Sandbox: %v", err)
	}
	defer func() {
		if _, err := sbFromID.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()
	fmt.Printf("Started first Sandbox: %s\n", sb.SandboxID)

	gitClone, err := sb.Exec(ctx, []string{
		"git",
		"clone",
		"https://github.com/modal-labs/libmodal.git",
		"/repo",
	}, nil)
	if err != nil {
		log.Fatalf("Failed to exec git clone: %v", err)
	}
	if exitCode, err := gitClone.Wait(ctx); err != nil || exitCode != 0 {
		log.Fatalf("Failed to wait for git clone: exit code: %d, err: %v", exitCode, err)
	}

	repoSnapshot, err := sb.SnapshotDirectory(ctx, "/repo")
	if err != nil {
		log.Fatalf("Failed to snapshot directory: %v", err)
	}
	fmt.Printf("Took a snapshot of the /repo directory, Image ID: %s\n", repoSnapshot.ImageID)

	if _, err := sb.Terminate(ctx, nil); err != nil {
		log.Fatalf("Failed to terminate Sandbox: %v", err)
	}

	// Start a new Sandbox, and mount the repo directory:
	sb2, err := mc.Sandboxes.Create(ctx, app, baseImage, nil)
	if err != nil {
		log.Fatalf("Failed to create second Sandbox: %v", err)
	}
	sb2FromID, err := mc.Sandboxes.FromID(ctx, sb2.SandboxID)
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer func() {
		if _, err := sb2FromID.Terminate(context.Background(), nil); err != nil {
			log.Fatalf("Failed to terminate Sandbox %s: %v", sb2.SandboxID, err)
		}
	}()
	fmt.Printf("Started second Sandbox: %s\n", sb2.SandboxID)

	mkdirProc2, err := sb2.Exec(ctx, []string{"mkdir", "-p", "/repo"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec mkdir in sb2: %v", err)
	}
	if exitCode, err := mkdirProc2.Wait(ctx); err != nil || exitCode != 0 {
		log.Fatalf("Failed to wait for mkdir in sb2: exit code: %d, err: %v", exitCode, err)
	}
	if err := sb2.MountImage(ctx, "/repo", repoSnapshot); err != nil {
		log.Fatalf("Failed to mount snapshot in sb2: %v", err)
	}

	repoLs, err := sb2.Exec(ctx, []string{"ls", "/repo"}, nil)
	if err != nil {
		log.Fatalf("Failed to exec ls: %v", err)
	}
	if exitCode, err := repoLs.Wait(ctx); err != nil || exitCode != 0 {
		log.Fatalf("Failed to wait for ls: exit code: %d, err: %v", exitCode, err)
	}
	output, err := io.ReadAll(repoLs.Stdout)
	if err != nil {
		log.Fatalf("Failed to read stdout: %v", err)
	}
	fmt.Printf("Contents of /repo directory in new Sandbox sb2:\n%s", output)

	// You can also optionally unmount the Image
	if err := sb2.UnmountImage(ctx, "/repo"); err != nil {
		log.Fatalf("Failed to unmount snapshot in sb2: %v", err)
	}
	fmt.Println("Unmounted the snapshot from /repo")

	if _, err := sb2.Terminate(ctx, nil); err != nil {
		log.Fatalf("Failed to terminate sb2: %v", err)
	}

	if err := mc.Images.Delete(ctx, repoSnapshot.ImageID, nil); err != nil {
		log.Fatalf("Failed to delete snapshot image: %v", err)
	}
}
