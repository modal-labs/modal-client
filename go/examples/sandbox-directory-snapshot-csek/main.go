// This example demonstrates customer-supplied encryption keys (CSEK) for
// Sandbox directory snapshots.
//
// Create and restore a CSEK-encrypted directory snapshot:
//
//	go run ./examples/sandbox-directory-snapshot-csek
//
// Restore an existing CSEK-encrypted directory snapshot:
//
//	go run ./examples/sandbox-directory-snapshot-csek \
//	  -image-id=im-... \
//	  -encryption-key=...
//
// You may pass -encryption-key to use your own base64-encoded key.
// If omitted, this example generates one and prints it so you can store it.

package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"flag"
	"fmt"
	"io"
	"log"
	"strings"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	imageID := flag.String("image-id", "", "snapshot Image ID to restore; omitted means create and restore a new snapshot")
	encryptionKey := flag.String("encryption-key", "", "base64-encoded customer-supplied encryption key")
	flag.Parse()

	ctx := context.Background()
	mc, err := modal.NewClient()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	app, err := mc.Apps.FromName(ctx, "libmodal-example", &modal.AppFromNameParams{CreateIfMissing: true})
	if err != nil {
		log.Fatalf("Failed to get or create App: %v", err)
	}

	if *imageID != "" {
		key := decodeKey(*encryptionKey)
		restoreSnapshot(ctx, mc, app, *imageID, key)
		return
	}

	key := decodeOrGenerateKey(*encryptionKey)
	snapshot := takeSnapshot(ctx, mc, app, key)
	restoreImage(ctx, mc, app, snapshot, key)
}

func decodeOrGenerateKey(keyBase64 string) []byte {
	if keyBase64 != "" {
		return decodeKey(keyBase64)
	}
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		log.Fatalf("Failed to generate encryption key: %v", err)
	}
	return key
}

func decodeKey(keyBase64 string) []byte {
	if keyBase64 == "" {
		log.Fatal("Set -encryption-key")
	}
	key, err := base64.StdEncoding.DecodeString(keyBase64)
	if err != nil {
		log.Fatalf("Failed to decode -encryption-key: %v", err)
	}
	return key
}

func takeSnapshot(ctx context.Context, mc *modal.Client, app *modal.App, encryptionKey []byte) *modal.Image {
	image := mc.Images.FromRegistry("alpine:3.21", nil)
	sb, err := mc.Sandboxes.Create(ctx, app, image, nil)
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer terminateSandbox(ctx, sb)

	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)
	if err := run(ctx, sb, []string{"sh", "-c", "mkdir -p /project && echo 'private data' > /project/state.txt"}); err != nil {
		log.Fatalf("Failed to write snapshot contents: %v", err)
	}

	snapshot, err := sb.SnapshotDirectory(ctx, "/project", &modal.SandboxSnapshotDirectoryParams{
		ExperimentalEncryptionKey: encryptionKey,
	})
	if err != nil {
		log.Fatalf("Failed to snapshot directory: %v", err)
	}

	fmt.Printf("Snapshot Image ID: %s\n", snapshot.ImageID)
	fmt.Printf("Encryption key (base64): %s\n", base64.StdEncoding.EncodeToString(encryptionKey))
	return snapshot
}

func restoreSnapshot(ctx context.Context, mc *modal.Client, app *modal.App, imageID string, encryptionKey []byte) {
	snapshot, err := mc.Images.FromID(ctx, imageID, nil)
	if err != nil {
		log.Fatalf("Failed to load snapshot Image: %v", err)
	}
	restoreImage(ctx, mc, app, snapshot, encryptionKey)
}

func restoreImage(ctx context.Context, mc *modal.Client, app *modal.App, snapshot *modal.Image, encryptionKey []byte) {
	image := mc.Images.FromRegistry("alpine:3.21", nil)
	sb, err := mc.Sandboxes.Create(ctx, app, image, nil)
	if err != nil {
		log.Fatalf("Failed to create Sandbox: %v", err)
	}
	defer terminateSandbox(ctx, sb)

	fmt.Printf("Started Sandbox: %s\n", sb.SandboxID)
	if err := run(ctx, sb, []string{"mkdir", "-p", "/project"}); err != nil {
		log.Fatalf("Failed to create mount directory: %v", err)
	}
	if err := sb.MountImage(ctx, "/project", snapshot, &modal.SandboxMountImageParams{
		ExperimentalEncryptionKey: encryptionKey,
	}); err != nil {
		log.Fatalf("Failed to mount snapshot: %v", err)
	}

	output, err := execOutput(ctx, sb, []string{"cat", "/project/state.txt"})
	if err != nil {
		log.Fatalf("Failed to read restored file: %v", err)
	}
	fmt.Printf("Restored file contents: %s\n", strings.TrimSpace(output))
}

func run(ctx context.Context, sb *modal.Sandbox, command []string) error {
	process, err := sb.Exec(ctx, command, nil)
	if err != nil {
		return fmt.Errorf("exec: %w", err)
	}
	exitCode, err := process.Wait(ctx, nil)
	if err != nil {
		return fmt.Errorf("wait: %w", err)
	}
	if exitCode != 0 {
		return fmt.Errorf("%q exited %d", command[0], exitCode)
	}
	return nil
}

func execOutput(ctx context.Context, sb *modal.Sandbox, command []string) (string, error) {
	process, err := sb.Exec(ctx, command, nil)
	if err != nil {
		return "", fmt.Errorf("exec: %w", err)
	}
	output, readErr := io.ReadAll(process.Stdout)
	exitCode, waitErr := process.Wait(ctx, nil)
	if waitErr != nil {
		return "", fmt.Errorf("wait: %w", waitErr)
	}
	if readErr != nil {
		return "", fmt.Errorf("read stdout: %w", readErr)
	}
	if exitCode != 0 {
		return "", fmt.Errorf("%q exited %d", command[0], exitCode)
	}
	return string(output), nil
}

func terminateSandbox(ctx context.Context, sb *modal.Sandbox) {
	if _, err := sb.Terminate(ctx, nil); err != nil {
		log.Printf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
	}
}
