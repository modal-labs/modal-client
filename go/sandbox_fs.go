package modal

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

const sandboxFsToolsPath = "/__modal/.bin/modal-sandbox-fs-tools"

// writeErrTracker wraps an io.Writer and records the most recent write error,
// so callers can distinguish write-side errors from read-side errors after io.Copy.
type writeErrTracker struct {
	w        io.Writer
	writeErr error
}

func (wt *writeErrTracker) Write(p []byte) (int, error) {
	n, err := wt.w.Write(p)
	if err != nil {
		wt.writeErr = err
	}
	return n, err
}

const taskCommandRouterMaxBufferSize = 16 * 1024 * 1024

// FileType represents the type of a filesystem entry.
type FileType string

const (
	FileTypeFile      FileType = "file"
	FileTypeDirectory FileType = "directory"
	FileTypeSymlink   FileType = "symlink"
)

// FileInfo holds metadata for a file, directory, or symlink in a Sandbox.
type FileInfo struct {
	Name          string   `json:"name"`
	Path          string   `json:"path"`
	Type          FileType `json:"type"`
	Size          int64    `json:"size"`
	Mode          uint32   `json:"mode"`
	Permissions   string   `json:"permissions"`
	Owner         string   `json:"owner"`
	Group         string   `json:"group"`
	ModifiedTime  float64  `json:"modified_time"`
	SymlinkTarget *string  `json:"symlink_target"`
}

// This is the narrow exec interface SandboxFilesystem needs, and is satisfied
// directly by *Sandbox in production or by panicSandbox for unit tests.
type sandboxForFilesystem interface {
	Exec(ctx context.Context, command []string, params *SandboxExecParams) (*ContainerProcess, error)
}

// SandboxFilesystem provides high-level filesystem operations for a running Sandbox.
// Obtain it via [Sandbox.Filesystem].
type SandboxFilesystem struct {
	sandbox sandboxForFilesystem
	logger  *slog.Logger
}

// SandboxFilesystemMakeDirectoryParams holds optional parameters for [SandboxFilesystem.MakeDirectory].
type SandboxFilesystemMakeDirectoryParams struct {
	// CreateParents controls whether missing parent directories are created automatically.
	// Defaults to true when nil.
	CreateParents *bool
}

// SandboxFilesystemRemoveParams holds optional parameters for [SandboxFilesystem.Remove].
type SandboxFilesystemRemoveParams struct {
	// Recurisve controls whether contens of a removed directory are recursively removed.
	// Defaults to false when nil.
	Recursive bool
}

// SandboxFilesystemCopyFromLocalParams holds optional parameters for [SandboxFilesystem.CopyFromLocal].
type SandboxFilesystemCopyFromLocalParams struct{}

// SandboxFilesystemCopyToLocalParams holds optional parameters for [SandboxFilesystem.CopyToLocal].
type SandboxFilesystemCopyToLocalParams struct{}

// SandboxFilesystemListFilesParams holds optional parameters for [SandboxFilesystem.ListFiles].
type SandboxFilesystemListFilesParams struct{}

// SandboxFilesystemReadParams holds optional parameters for [SandboxFilesystem.ReadBytes] and [SandboxFilesystem.ReadText].
type SandboxFilesystemReadParams struct{}

// SandboxFilesystemStatParams holds optional parameters for [SandboxFilesystem.Stat].
type SandboxFilesystemStatParams struct{}

// SandboxFilesystemWriteParams holds optional parameters for [SandboxFilesystem.WriteBytes] and [SandboxFilesystem.WriteText].
type SandboxFilesystemWriteParams struct{}

// CopyFromLocal copies a local file into the Sandbox.
//
// remotePath must be an absolute path to a file in the Sandbox.
// Parent directories are created if needed. The remote file is overwritten
// if it already exists.
//
// Returns [SandboxFilesystemNotADirectoryError] if a parent component of
// remotePath is not a directory, [SandboxFilesystemIsADirectoryError] if
// remotePath points to a directory, [SandboxFilesystemPermissionError] if
// write permission is denied, or an *os.PathError if localPath does not
// exist, is a directory, or cannot be read.
func (fsys *SandboxFilesystem) CopyFromLocal(ctx context.Context, localPath, remotePath string, params *SandboxFilesystemCopyFromLocalParams) error {
	if err := validateAbsoluteRemotePath(remotePath, "CopyFromLocal"); err != nil {
		return err
	}

	f, err := os.Open(localPath)
	if err != nil {
		return err
	}
	defer func() {
		if err := f.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "CopyFromLocal: close file", "error", err)
		}
	}()

	execCtx, cancelExec := context.WithCancel(ctx)
	defer cancelExec()

	cp, err := fsys.sandbox.Exec(execCtx, []string{sandboxFsToolsPath, makeWriteFileCommand(remotePath)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "CopyFromLocal", remotePath, err)
	}

	earlyExit := false
	buf := make([]byte, taskCommandRouterMaxBufferSize)
	for {
		n, readErr := f.Read(buf)
		if n > 0 {
			if _, werr := cp.Stdin.Write(buf[:n]); werr != nil {
				if isBinaryExitedEarly(werr) {
					earlyExit = true
					break
				}
				return translateExecError(ctx, fsys.logger, "CopyFromLocal", remotePath, werr)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			// Cancel the exec rather than closing stdin cleanly. A clean close
			// sends EOF, which the remote process interprets as "write complete"
			// and would create an empty (or partial) file despite the local error.
			cancelExec()
			return readErr
		}
	}

	if cerr := cp.Stdin.Close(); cerr != nil {
		if earlyExit {
			fsys.logger.DebugContext(ctx, "CopyFromLocal: close stdin", "error", cerr)
		} else if !isBinaryExitedEarly(cerr) {
			return cerr
		}
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "CopyFromLocal: close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(execCtx)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "CopyFromLocal", remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, "CopyFromLocal: read stderr", "error", err)
		}
		return raiseWriteFileError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	return nil
}

// CopyToLocal copies a file from the Sandbox to a local path.
//
// remotePath must be an absolute path to a file in the Sandbox.
// Parent directories for localPath are created if needed. The local file is
// overwritten if it already exists.
//
// Returns [SandboxFilesystemNotFoundError] if the remote path does not exist,
// [SandboxFilesystemIsADirectoryError] if the remote path points to a directory,
// [SandboxFilesystemFileTooLargeError] if the file exceeds the read size limit,
// or [SandboxFilesystemPermissionError] if read permission is denied.
func (fsys *SandboxFilesystem) CopyToLocal(ctx context.Context, remotePath, localPath string, params *SandboxFilesystemCopyToLocalParams) (retErr error) {
	if err := validateAbsoluteRemotePath(remotePath, "CopyToLocal"); err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return err
	}

	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeReadFileCommand(remotePath)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "CopyToLocal", remotePath, err)
	}

	f, err := os.CreateTemp(filepath.Dir(localPath), ".modal-sandbox-fs-tmp-*")
	if err != nil {
		return err
	}
	defer func() {
		// Only clean up on failure; on success the file is closed and renamed.
		if retErr != nil {
			if err := f.Close(); err != nil {
				fsys.logger.DebugContext(ctx, "CopyToLocal: close temp file", "error", err)
			}
			if err := os.Remove(f.Name()); err != nil {
				fsys.logger.DebugContext(ctx, "CopyToLocal: remove temp file", "error", err)
			}
		}
	}()

	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "CopyToLocal", remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, "CopyToLocal: read stderr", "error", err)
		}
		return raiseReadFileError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	wt := &writeErrTracker{w: f}
	if _, err := io.Copy(wt, cp.Stdout); err != nil {
		if wt.writeErr != nil {
			return wt.writeErr
		}
		return translateExecError(ctx, fsys.logger, "CopyToLocal", remotePath, err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), localPath)
}

// ListFiles lists files and directories in a Sandbox directory.
//
// remotePath must be an absolute path to a directory in the Sandbox.
// Returns a slice of [FileInfo] objects sorted by name.
//
// Returns [SandboxFilesystemNotFoundError] if the path does not exist,
// [SandboxFilesystemNotADirectoryError] if the path is not a directory,
// or [SandboxFilesystemPermissionError] if read permission is denied.
func (fsys *SandboxFilesystem) ListFiles(ctx context.Context, remotePath string, params *SandboxFilesystemListFilesParams) ([]FileInfo, error) {
	if err := validateAbsoluteRemotePath(remotePath, "ListFiles"); err != nil {
		return nil, err
	}
	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeListFilesCommand(remotePath)}, nil)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "ListFiles", remotePath, err)
	}

	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "ListFiles", remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, "ListFiles: read stderr", "error", err)
		}
		return nil, raiseListFilesError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	stdout, err := io.ReadAll(cp.Stdout)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "ListFiles", remotePath, err)
	}
	var entries []FileInfo
	if err := json.Unmarshal(stdout, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

// MakeDirectory creates a new directory in the Sandbox.
//
// remotePath must be an absolute path in the Sandbox.
//
// When params.CreateParents is true (the default when params is nil), any
// missing parent directories are created and the call is idempotent (succeeds
// if the directory already exists). When false, the immediate parent must
// already exist and the path must not already exist.
//
// Returns [SandboxFilesystemNotFoundError] if the parent does not exist and
// CreateParents is false, [SandboxFilesystemPathAlreadyExistsError] if the
// path already exists, [SandboxFilesystemNotADirectoryError] if a path
// component is not a directory, [SandboxFilesystemPermissionError] if
// creation is not permitted, or [InvalidError] if the mount does not
// support this operation.
func (fsys *SandboxFilesystem) MakeDirectory(ctx context.Context, remotePath string, params *SandboxFilesystemMakeDirectoryParams) error {
	if err := validateAbsoluteRemotePath(remotePath, "MakeDirectory"); err != nil {
		return err
	}
	createParents := params == nil || params.CreateParents == nil || *params.CreateParents
	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeMakeDirectoryCommand(remotePath, createParents)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "MakeDirectory", remotePath, err)
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "MakeDirectory: close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "MakeDirectory", remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, "MakeDirectory: read stderr", "error", err)
		}
		return raiseMakeDirectoryError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	return nil
}

func (fsys *SandboxFilesystem) readFile(ctx context.Context, operation, remotePath string, _ *SandboxFilesystemReadParams) ([]byte, error) {
	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeReadFileCommand(remotePath)}, nil)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}

	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, operation+": read stderr", "error", err)
		}
		return nil, raiseReadFileError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	stdout, err := io.ReadAll(cp.Stdout)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}
	return stdout, nil
}

// ReadBytes reads a file from the Sandbox and returns its contents as bytes.
//
// remotePath must be an absolute path to a file in the Sandbox.
//
// Returns [SandboxFilesystemNotFoundError] if the path does not exist,
// [SandboxFilesystemIsADirectoryError] if the path points to a directory,
// [SandboxFilesystemFileTooLargeError] if the file exceeds the read size limit,
// or [SandboxFilesystemPermissionError] if read permission is denied.
func (fsys *SandboxFilesystem) ReadBytes(ctx context.Context, remotePath string, params *SandboxFilesystemReadParams) ([]byte, error) {
	if err := validateAbsoluteRemotePath(remotePath, "ReadBytes"); err != nil {
		return nil, err
	}
	return fsys.readFile(ctx, "ReadBytes", remotePath, params)
}

// ReadText reads a file from the Sandbox and returns its contents as a UTF-8 string.
//
// remotePath must be an absolute path to a file in the Sandbox.
//
// Returns [SandboxFilesystemNotFoundError] if the path does not exist,
// [SandboxFilesystemIsADirectoryError] if the path points to a directory,
// [SandboxFilesystemFileTooLargeError] if the file exceeds the read size limit,
// or [SandboxFilesystemPermissionError] if read permission is denied.
func (fsys *SandboxFilesystem) ReadText(ctx context.Context, remotePath string, params *SandboxFilesystemReadParams) (string, error) {
	if err := validateAbsoluteRemotePath(remotePath, "ReadText"); err != nil {
		return "", err
	}
	stdout, err := fsys.readFile(ctx, "ReadText", remotePath, params)
	return string(stdout), err
}

// Remove a file or directory in the Sandbox.
//
// remotePath must be an absolute path in the Sandbox. When remotePath is a
// directory and params.Recursive is false (the default when params is nil),
// it is removed only if empty. When Recursive is true, the directory and all
// its contents are removed. Recursive removal is not supported on all mounts.
//
// Returns [SandboxFilesystemNotFoundError] if the path does not exist,
// [SandboxFilesystemDirectoryNotEmptyError] if Recursive is false and the
// directory is not empty, [SandboxFilesystemPermissionError] if removal is
// not permitted, or [InvalidError] if the mount does not support this operation.
func (fsys *SandboxFilesystem) Remove(ctx context.Context, remotePath string, params *SandboxFilesystemRemoveParams) error {
	if err := validateAbsoluteRemotePath(remotePath, "Remove"); err != nil {
		return err
	}
	if params == nil {
		params = &SandboxFilesystemRemoveParams{}
	}
	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeRemoveCommand(remotePath, params.Recursive)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "Remove", remotePath, err)
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "Remove: close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "Remove", remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, "Remove: read stderr", "error", err)
		}
		return raiseRemoveError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	return nil
}

// Stat returns metadata for a single file, directory, or symlink in the Sandbox.
//
// remotePath must be an absolute path in the Sandbox. If remotePath is a
// symlink, the returned [FileInfo] describes the symlink itself, not the
// target it points to.
//
// Returns [SandboxFilesystemNotFoundError] if the path does not exist,
// [SandboxFilesystemNotADirectoryError] if a non-leaf component of the path
// is not a directory, or [SandboxFilesystemPermissionError] if a path
// component is not searchable.
func (fsys *SandboxFilesystem) Stat(ctx context.Context, remotePath string, params *SandboxFilesystemStatParams) (*FileInfo, error) {
	if err := validateAbsoluteRemotePath(remotePath, "Stat"); err != nil {
		return nil, err
	}
	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeStatCommand(remotePath)}, nil)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "Stat", remotePath, err)
	}

	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "Stat", remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, "Stat: read stderr", "error", err)
		}
		return nil, raiseStatError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	stdout, err := io.ReadAll(cp.Stdout)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "Stat", remotePath, err)
	}
	var entry FileInfo
	if err := json.Unmarshal(stdout, &entry); err != nil {
		return nil, err
	}
	return &entry, nil
}

func (fsys *SandboxFilesystem) writeFile(ctx context.Context, operation string, data []byte, remotePath string, _ *SandboxFilesystemWriteParams) error {
	cp, err := fsys.sandbox.Exec(ctx, []string{sandboxFsToolsPath, makeWriteFileCommand(remotePath)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}

	earlyExit := false
	// max(len(data), 1) ensures the loop runs at least once when data is empty,
	// which is required to create an empty file via a single stdin write.
	for offset := 0; offset < max(len(data), 1); offset += taskCommandRouterMaxBufferSize {
		chunk := data[offset:min(offset+taskCommandRouterMaxBufferSize, len(data))]
		if _, werr := cp.Stdin.Write(chunk); werr != nil {
			if isBinaryExitedEarly(werr) {
				earlyExit = true
				break
			}
			return translateExecError(ctx, fsys.logger, operation, remotePath, werr)
		}
	}
	if cerr := cp.Stdin.Close(); cerr != nil {
		if earlyExit {
			fsys.logger.DebugContext(ctx, operation+": close stdin", "error", cerr)
		} else if !isBinaryExitedEarly(cerr) {
			return cerr
		}
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, operation+": close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(ctx)
	if err != nil {
		return translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}
	if returnCode != 0 {
		stderr, err := io.ReadAll(cp.Stderr)
		if err != nil {
			fsys.logger.DebugContext(ctx, operation+": read stderr", "error", err)
		}
		return raiseWriteFileError(ctx, fsys.logger, returnCode, stderr, remotePath)
	}
	return nil
}

// WriteBytes writes binary content to a file in the Sandbox.
//
// remotePath must be an absolute path to a file in the Sandbox.
// Parent directories are created if needed. The remote file is overwritten
// if it already exists.
//
// Returns [SandboxFilesystemNotADirectoryError] if a parent component of
// remotePath is not a directory, [SandboxFilesystemIsADirectoryError] if
// remotePath points to a directory, or [SandboxFilesystemPermissionError]
// if write permission is denied.
func (fsys *SandboxFilesystem) WriteBytes(ctx context.Context, data []byte, remotePath string, params *SandboxFilesystemWriteParams) error {
	if err := validateAbsoluteRemotePath(remotePath, "WriteBytes"); err != nil {
		return err
	}
	return fsys.writeFile(ctx, "WriteBytes", data, remotePath, params)
}

// WriteText writes UTF-8 text to a file in the Sandbox.
//
// remotePath must be an absolute path to a file in the Sandbox.
// Parent directories are created if needed. The remote file is overwritten
// if it already exists.
//
// Returns [SandboxFilesystemNotADirectoryError] if a parent component of
// remotePath is not a directory, [SandboxFilesystemIsADirectoryError] if
// remotePath points to a directory, or [SandboxFilesystemPermissionError]
// if write permission is denied.
func (fsys *SandboxFilesystem) WriteText(ctx context.Context, data string, remotePath string, params *SandboxFilesystemWriteParams) error {
	if err := validateAbsoluteRemotePath(remotePath, "WriteText"); err != nil {
		return err
	}
	return fsys.writeFile(ctx, "WriteText", []byte(data), remotePath, params)
}
