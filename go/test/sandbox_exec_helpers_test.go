package test

import (
	"context"
	"fmt"
	"io"

	modal "github.com/modal-labs/modal-client/go"
)

func writeRemoteFile(ctx context.Context, sb *modal.Sandbox, path string, data []byte) error {
	cmd := `mkdir -p "$(dirname '` + path + `')" && cat > '` + path + `'`
	cp, err := sb.Exec(ctx, []string{"sh", "-c", cmd}, nil)
	if err != nil {
		return err
	}
	defer func() { _ = cp.Stdout.Close() }()
	defer func() { _ = cp.Stderr.Close() }()
	if _, err := cp.Stdin.Write(data); err != nil {
		_ = cp.Stdin.Close()
		_, _ = cp.Wait(ctx, nil)
		return fmt.Errorf("writeRemoteFile write: %w", err)
	}
	_ = cp.Stdin.Close()
	rc, err := cp.Wait(ctx, nil)
	if err != nil {
		return err
	}
	if rc != 0 {
		return fmt.Errorf("writeRemoteFile: %s exited %d", path, rc)
	}
	return nil
}

func readRemoteFile(ctx context.Context, sb *modal.Sandbox, path string) ([]byte, error) {
	cp, err := sb.Exec(ctx, []string{"cat", path}, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = cp.Stdout.Close() }()
	defer func() { _ = cp.Stderr.Close() }()

	type readResult struct {
		stdout []byte
		err    error
	}
	readDone := make(chan readResult, 1)
	go func() {
		stdout, err := io.ReadAll(cp.Stdout)
		readDone <- readResult{stdout: stdout, err: err}
	}()

	rc, err := cp.Wait(ctx, nil)
	result := <-readDone
	if err != nil {
		return nil, err
	}
	if result.err != nil {
		return nil, fmt.Errorf("readRemoteFile read: %w", result.err)
	}
	if rc != 0 {
		return nil, fmt.Errorf("readRemoteFile: %s exited %d", path, rc)
	}
	return result.stdout, nil
}
