// Example demonstrating the Sandbox.Filesystem() namespace API.
//
// This example shows how to:
//   - Write and read text files
//   - Write and read binary files
//   - Inspect file metadata
//   - Create directories and list their contents
//   - Upload and download files between local disk and the Sandbox
//   - Delete files and directories
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

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
			log.Printf("Failed to terminate Sandbox %s: %v", sb.SandboxID, err)
		}
	}()

	fs := sb.Filesystem()

	// ── write & read text ─────────────────────────────────────────────────

	if err := fs.WriteText(ctx, "Hello from Sandbox.Filesystem()!\n", "/tmp/hello.txt", nil); err != nil {
		log.Fatalf("WriteText: %v", err)
	}

	text, err := fs.ReadText(ctx, "/tmp/hello.txt", nil)
	if err != nil {
		log.Fatalf("ReadText: %v", err)
	}
	fmt.Printf("ReadText: %s", text)

	// ── write & read bytes ────────────────────────────────────────────────

	payload := []byte{0x48, 0x65, 0x6c, 0x6c, 0x6f} // "Hello"
	if err := fs.WriteBytes(ctx, payload, "/tmp/hello.bin", nil); err != nil {
		log.Fatalf("WriteBytes: %v", err)
	}

	bytes, err := fs.ReadBytes(ctx, "/tmp/hello.bin", nil)
	if err != nil {
		log.Fatalf("ReadBytes: %v", err)
	}
	fmt.Printf("ReadBytes: %v\n", bytes)

	// ── stat ──────────────────────────────────────────────────────────────

	info, err := fs.Stat(ctx, "/tmp/hello.txt", nil)
	if err != nil {
		log.Fatalf("Stat: %v", err)
	}
	fmt.Printf("Stat: name=%s, size=%d, permissions=%s\n", info.Name, info.Size, info.Permissions)

	// ── make_directory & list_files ───────────────────────────────────────

	if err := fs.MakeDirectory(ctx, "/tmp/mydir/nested", nil); err != nil {
		log.Fatalf("MakeDirectory: %v", err)
	}
	if err := fs.WriteText(ctx, "nested file\n", "/tmp/mydir/nested/file.txt", nil); err != nil {
		log.Fatalf("WriteText nested: %v", err)
	}
	if err := fs.WriteText(ctx, "top-level file\n", "/tmp/mydir/top.txt", nil); err != nil {
		log.Fatalf("WriteText top: %v", err)
	}

	entries, err := fs.ListFiles(ctx, "/tmp/mydir", nil)
	if err != nil {
		log.Fatalf("ListFiles: %v", err)
	}
	fmt.Println("ListFiles /tmp/mydir:")
	for _, e := range entries {
		fmt.Printf("  %s  (%s)\n", e.Name, e.Type)
	}

	// ── copy_from_local & copy_to_local ───────────────────────────────────

	tmpDir, err := os.MkdirTemp("", "modal-fs-example-*")
	if err != nil {
		log.Fatalf("MkdirTemp: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			log.Printf("RemoveAll %s: %v", tmpDir, err)
		}
	}()

	localSrc := filepath.Join(tmpDir, "upload.txt")
	if err := os.WriteFile(localSrc, []byte("Uploaded via CopyFromLocal\n"), 0o644); err != nil {
		log.Fatalf("WriteFile: %v", err)
	}

	if err := fs.CopyFromLocal(ctx, localSrc, "/tmp/uploaded.txt", nil); err != nil {
		log.Fatalf("CopyFromLocal: %v", err)
	}

	localDst := filepath.Join(tmpDir, "download.txt")
	if err := fs.CopyToLocal(ctx, "/tmp/uploaded.txt", localDst, nil); err != nil {
		log.Fatalf("CopyToLocal: %v", err)
	}

	downloaded, err := os.ReadFile(localDst)
	if err != nil {
		log.Fatalf("ReadFile: %v", err)
	}
	fmt.Printf("Copy round-trip: %s", downloaded)

	// ── remove ────────────────────────────────────────────────────────────

	if err := fs.Remove(ctx, "/tmp/hello.bin", nil); err != nil {
		log.Fatalf("Remove: %v", err)
	}
	if err := fs.Remove(ctx, "/tmp/mydir", &modal.SandboxFilesystemRemoveParams{Recursive: true}); err != nil {
		log.Fatalf("Remove recursive: %v", err)
	}
	fmt.Println("Remove: cleaned up")
}
