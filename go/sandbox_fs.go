package modal

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"time"
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

// readErrTracker wraps an io.ReadSeeker and records the most recent non-EOF
// read error, so callers can distinguish local read errors from stream errors.
type readErrTracker struct {
	rs      io.ReadSeeker
	readErr error
}

func (rt *readErrTracker) Read(p []byte) (int, error) {
	n, err := rt.rs.Read(p)
	if err != nil && err != io.EOF {
		rt.readErr = err
	}
	return n, err
}

func (rt *readErrTracker) Seek(offset int64, whence int) (int64, error) {
	return rt.rs.Seek(offset, whence)
}

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

// FileWatchEventType is the category of a filesystem change event.
type FileWatchEventType string

const (
	FileWatchEventTypeAccess  FileWatchEventType = "Access"
	FileWatchEventTypeCreate  FileWatchEventType = "Create"
	FileWatchEventTypeModify  FileWatchEventType = "Modify"
	FileWatchEventTypeRemove  FileWatchEventType = "Remove"
	FileWatchEventTypeUnknown FileWatchEventType = "Unknown"
)

// FileWatchEvent is a single filesystem change reported by Watch.
//
// Paths contains the absolute path(s) affected by the event. For most event
// types it holds a single entry. Rename operations are reported as Modify
// events: when both the source and destination fall within the watched,
// scope, Paths holds [source, destination]; when only one side of the rename
// is visible, Paths holds that single path.
type FileWatchEvent struct {
	EventType FileWatchEventType
	Paths     []string
}

// Rust rename variants that all collapse to FileWatchEventTypeModify.
var rustRenameVariants = []string{"Rename", "RenameFrom", "RenameTo"}

// Expand a user-facing filter to the string to modal-sandbox-fs-tools event type strings.
func expandWatchFilter(filter []FileWatchEventType) []string {
	result := make([]string, 0, len(filter)+len(rustRenameVariants))
	for _, t := range filter {
		result = append(result, string(t))
		if t == FileWatchEventTypeModify {
			result = append(result, rustRenameVariants...)
		}
	}
	return result
}

// This is the narrow exec interface SandboxFilesystem needs, satisfied by
// *Sandbox and *SidecarContainer in production via an unexported adapter
// method, or by panicSandbox in unit tests. The method is unexported so that
// the two public Exec surfaces are free to take different param types.
type sandboxForFilesystem interface {
	execForFilesystem(ctx context.Context, command []string, params *SandboxExecParams) (*ContainerProcess, error)
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

// SandboxFilesystemWatchParams holds optional parameters for [SandboxFilesystem.Watch].
type SandboxFilesystemWatchParams struct {
	Filter    []FileWatchEventType
	Recursive bool
	// Timeout is the maximum duration to watch. A nil Timeout watches
	// indefinitely, while a zero Timeout returns immediately without waiting for
	// events. Durations are rounded-down to the nearest whole number of seconds.
	Timeout *time.Duration
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

	cp, err := fsys.sandbox.execForFilesystem(execCtx, []string{sandboxFsToolsPath, makeWriteFileCommand(remotePath)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "CopyFromLocal", remotePath, err)
	}

	src := &readErrTracker{rs: f}
	if _, werr := cp.stdinWriteStream(execCtx, src); werr != nil && !isBinaryExitedEarly(werr) {
		if src.readErr != nil {
			// Local read error: cancel the exec rather than closing stdin
			// cleanly, which would send EOF and write the file on the sandbox.
			cancelExec()
			return src.readErr
		}
		return translateExecError(ctx, fsys.logger, "CopyFromLocal", remotePath, werr)
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "CopyFromLocal: close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(execCtx, nil)
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

	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeReadFileCommand(remotePath)}, nil)
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

	returnCode, err := cp.Wait(ctx, nil)
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
	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeListFilesCommand(remotePath)}, nil)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "ListFiles", remotePath, err)
	}

	returnCode, err := cp.Wait(ctx, nil)
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
	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeMakeDirectoryCommand(remotePath, createParents)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "MakeDirectory", remotePath, err)
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "MakeDirectory: close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(ctx, nil)
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
	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeReadFileCommand(remotePath)}, nil)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}

	returnCode, err := cp.Wait(ctx, nil)
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
	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeRemoveCommand(remotePath, params.Recursive)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, "Remove", remotePath, err)
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, "Remove: close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(ctx, nil)
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
	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeStatCommand(remotePath)}, nil)
	if err != nil {
		return nil, translateExecError(ctx, fsys.logger, "Stat", remotePath, err)
	}

	returnCode, err := cp.Wait(ctx, nil)
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

// Watch a path in the Sandbox for filesystem changes.
//
// remotePath must be an absolute path in the Sandbox. If it points to a
// file, events for that file are reported. If it points to a directory,
// events for entries directly inside it are reported. Set params.Recursive
// to also receive events for all nested subdirectories. If remotePath is a
// symlink, it is followed and events reference paths under the resolved
// target.
//
// The returned [iter.Seq2] yields [FileWatchEvent] values as changes occur,
// until the timeout elapses, the caller breaks from the range loop, ctx is
// cancelled, or the Sandbox is terminated. The remote watch process is not
// started until iteration begins, so a sequence that is never ranged over
// launches nothing.
//
// Set params.Filter to restrict which event types are emitted. A nil filter
// permits all types; an empty slice suppresses all events.
//
// A nil params.Timeout watches indefinitely, while a zero params.Timeout
// returns immediately without waiting for events. Otherwise the duration is
// rounded down to whole seconds, and when it elapses the iterator stops
// without returning an error.
//
// Pass nil params for defaults (no filter, non-recursive, no timeout).
//
// Returns [SandboxFilesystemNotFoundError] if remotePath does not exist,
// [SandboxFilesystemPermissionError] if watch access is denied, or
// [InvalidError] if the filesystem does not support watching.
func (fsys *SandboxFilesystem) Watch(
	ctx context.Context,
	remotePath string,
	params *SandboxFilesystemWatchParams,
) (iter.Seq2[FileWatchEvent, error], error) {
	if err := validateAbsoluteRemotePath(remotePath, "Watch"); err != nil {
		return nil, err
	}
	if params == nil {
		params = &SandboxFilesystemWatchParams{}
	}

	var timeoutSecs *int
	if params.Timeout != nil {
		s := int(params.Timeout.Seconds())
		timeoutSecs = &s
	}

	var filterStrings []string
	if params.Filter != nil {
		filterStrings = expandWatchFilter(params.Filter)
	}

	// The remote watch process is launched inside the iterator, when iteration
	// begins, so its lifetime and timeout clock are bounded by consumption of
	// the returned sequence. A sequence that is never ranged over starts no
	// remote process.
	return func(yield func(FileWatchEvent, error) bool) {
		cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeWatchCommand(remotePath, params.Recursive, filterStrings, timeoutSecs)}, nil)
		if err != nil {
			yield(FileWatchEvent{}, translateExecError(ctx, fsys.logger, "Watch", remotePath, err))
			return
		}

		closeStdout := func() {
			if err := cp.Stdout.Close(); err != nil {
				fsys.logger.DebugContext(ctx, "Watch: close stdout", "error", err)
			}
		}
		// A cancelled ctx closes stdout, which unblocks the scanner below.
		defer context.AfterFunc(ctx, closeStdout)()
		defer closeStdout()

		// waitProcess closes stdin and waits for sandbox-fs-tools to
		// terminate.
		waitProcess := func() (int, error) {
			_ = cp.Stdin.Close()
			return cp.Wait(ctx, nil)
		}

		scanner := bufio.NewScanner(cp.Stdout)
		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}
			var raw struct {
				EventType string   `json:"event_type"`
				Paths     []string `json:"paths"`
			}
			if err := json.Unmarshal(line, &raw); err != nil {
				if !yield(FileWatchEvent{}, fmt.Errorf("parsing watch event: %w", err)) {
					_, _ = waitProcess()
					return
				}
				continue
			}
			if len(raw.Paths) == 0 {
				continue
			}
			rawType := raw.EventType
			if slices.Contains(rustRenameVariants, rawType) {
				rawType = "Modify"
			}
			if !yield(FileWatchEvent{EventType: FileWatchEventType(rawType), Paths: raw.Paths}, nil) {
				_, _ = waitProcess()
				return
			}
		}
		returnCode, waitErr := waitProcess()
		if err := scanner.Err(); err != nil {
			yield(FileWatchEvent{}, fmt.Errorf("reading watch events: %w", err))
			return
		}
		if waitErr != nil {
			yield(FileWatchEvent{}, translateExecError(ctx, fsys.logger, "Watch", remotePath, waitErr))
			return
		}
		if returnCode != 0 {
			stderr, err := io.ReadAll(cp.Stderr)
			if err != nil {
				fsys.logger.DebugContext(ctx, "Watch: read stderr", "error", err)
			}
			yield(FileWatchEvent{}, raiseWatchError(ctx, fsys.logger, returnCode, stderr, remotePath))
		}
	}, nil
}

func (fsys *SandboxFilesystem) writeFile(ctx context.Context, operation string, data []byte, remotePath string, _ *SandboxFilesystemWriteParams) error {
	cp, err := fsys.sandbox.execForFilesystem(ctx, []string{sandboxFsToolsPath, makeWriteFileCommand(remotePath)}, nil)
	if err != nil {
		return translateExecError(ctx, fsys.logger, operation, remotePath, err)
	}

	// Note empty data still creates an empty file.
	if _, werr := cp.stdinWriteStream(ctx, bytes.NewReader(data)); werr != nil && !isBinaryExitedEarly(werr) {
		return translateExecError(ctx, fsys.logger, operation, remotePath, werr)
	}

	defer func() {
		if err := cp.Stdout.Close(); err != nil {
			fsys.logger.DebugContext(ctx, operation+": close stdout", "error", err)
		}
	}()
	returnCode, err := cp.Wait(ctx, nil)
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
