package modal

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var supportErrorCodeRe = regexp.MustCompile(`Error code:\s*([A-Z0-9]{8})`)

type errorPayload struct {
	ErrorKind string `json:"error_kind"`
	Message   string `json:"message"`
	Detail    string `json:"detail"`
}

func tryParseErrorPayload(stderr []byte) *errorPayload {
	text := strings.TrimSpace(string(stderr))
	if text == "" {
		return nil
	}
	var p errorPayload
	if err := json.Unmarshal([]byte(text), &p); err != nil {
		return nil
	}
	if p.ErrorKind == "" || strings.TrimSpace(p.Message) == "" {
		return nil
	}
	return &p
}

func isSandboxUnavailable(err error) bool {
	if _, ok := status.FromError(err); !ok {
		// status.Code returns codes.Unknown for any non-gRPC error, so guard
		// against treating local OS errors (e.g. a failed local file write) as
		// "sandbox unavailable".
		return false
	}
	switch status.Code(err) {
	case codes.NotFound, codes.Canceled, codes.Unknown, codes.DeadlineExceeded, codes.Unavailable:
		return true
	}
	return false
}

func isBinaryExitedEarly(err error) bool {
	c := status.Code(err)
	return c == codes.FailedPrecondition || c == codes.Aborted
}

func extractSupportErrorCode(err error) string {
	if m := supportErrorCodeRe.FindStringSubmatch(err.Error()); len(m) > 1 {
		return m[1]
	}
	return ""
}

func translateExecError(ctx context.Context, log *slog.Logger, operation, remotePath string, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if isSandboxUnavailable(err) {
		log.DebugContext(ctx, "Sandbox filesystem control-plane error",
			"operation", operation,
			"path", remotePath,
			"error", err,
		)
		return NotFoundError{Exception: "The Sandbox is unavailable. This Sandbox may have already shut down."}
	}
	log.DebugContext(ctx, "Unexpected sandbox filesystem exec error",
		"operation", operation,
		"path", remotePath,
		"error", err,
	)
	if code := extractSupportErrorCode(err); code != "" {
		return SandboxFilesystemError{Exception: fmt.Sprintf("An unexpected error occurred, please contact support@modal.com (Error code: %s)", code)}
	}
	return SandboxFilesystemError{Exception: "An unexpected error occurred, please contact support@modal.com"}
}

func validateAbsoluteRemotePath(remotePath, operation string) error {
	if !strings.HasPrefix(remotePath, "/") {
		return InvalidError{Exception: fmt.Sprintf("Sandbox.filesystem.%s() currently only supports absolute remote_path values", operation)}
	}
	return nil
}

func raiseListFilesError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools list_files error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotFound":
			return SandboxFilesystemNotFoundError{Exception: p.Message + ": " + remotePath}
		case "IsFile", "NotDirectory":
			return SandboxFilesystemNotADirectoryError{Exception: p.Message + ": " + remotePath}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeListFilesCommand(remotePath string) string {
	b, _ := json.Marshal(map[string]any{"ListFiles": map[string]any{"path": remotePath}})
	return string(b)
}

func raiseMakeDirectoryError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools make_directory error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotFound":
			return SandboxFilesystemNotFoundError{Exception: p.Message + ": " + remotePath}
		case "PathAlreadyExists":
			return SandboxFilesystemPathAlreadyExistsError{Exception: p.Message + ": " + remotePath}
		case "NotDirectory":
			return SandboxFilesystemNotADirectoryError{Exception: p.Message + ": " + remotePath}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		case "NotSupported":
			return InvalidError{Exception: p.Message + ": " + remotePath + " - this operation is not supported for CloudBucketMounts"}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeMakeDirectoryCommand(remotePath string, createParents bool) string {
	b, _ := json.Marshal(map[string]any{"MakeDirectory": map[string]any{"path": remotePath, "parents": createParents}})
	return string(b)
}

func raiseReadFileError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools read error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotFound":
			return SandboxFilesystemNotFoundError{Exception: p.Message + ": " + remotePath}
		case "IsDirectory":
			return SandboxFilesystemIsADirectoryError{Exception: p.Message + ": " + remotePath}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		case "FileTooLarge":
			return SandboxFilesystemFileTooLargeError{Exception: p.Message + ": " + remotePath}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeReadFileCommand(remotePath string) string {
	b, _ := json.Marshal(map[string]any{"ReadFile": map[string]any{"path": remotePath}})
	return string(b)
}

func raiseRemoveError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools remove error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotFound":
			return SandboxFilesystemNotFoundError{Exception: p.Message + ": " + remotePath}
		case "DirectoryNotEmpty":
			return SandboxFilesystemDirectoryNotEmptyError{Exception: p.Message + ": " + remotePath}
		case "NotSupported":
			return InvalidError{Exception: p.Message + ": " + remotePath + " - this operation is not supported for CloudBucketMounts"}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeRemoveCommand(remotePath string, recursive bool) string {
	b, _ := json.Marshal(map[string]any{"Remove": map[string]any{"path": remotePath, "recursive": recursive}})
	return string(b)
}

func raiseStatError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools stat error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotFound":
			return SandboxFilesystemNotFoundError{Exception: p.Message + ": " + remotePath}
		case "NotDirectory":
			return SandboxFilesystemNotADirectoryError{Exception: p.Message + ": " + remotePath}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeStatCommand(remotePath string) string {
	b, _ := json.Marshal(map[string]any{"Stat": map[string]any{"path": remotePath}})
	return string(b)
}

func raiseWatchError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools watch error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotFound":
			return SandboxFilesystemNotFoundError{Exception: p.Message + ": " + remotePath}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		case "NotSupported":
			return InvalidError{Exception: p.Message + ": " + remotePath + " - this operation is not supported for CloudBucketMounts"}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeWatchCommand(remotePath string, recursive bool, filter []string, timeoutSecs *int) string {
	var filterVal any
	if filter != nil {
		filterVal = filter
	}
	var timeoutVal any
	if timeoutSecs != nil {
		timeoutVal = *timeoutSecs
	}
	b, _ := json.Marshal(map[string]any{"Watch": map[string]any{"path": remotePath, "recursive": recursive, "filter": filterVal, "timeout_secs": timeoutVal}})
	return string(b)
}

func raiseWriteFileError(ctx context.Context, log *slog.Logger, returnCode int, stderr []byte, remotePath string) error {
	if p := tryParseErrorPayload(stderr); p != nil {
		log.DebugContext(ctx, "sandbox-fs-tools write error",
			"path", remotePath,
			"error_kind", p.ErrorKind,
			"message", p.Message,
			"detail", p.Detail,
		)
		switch p.ErrorKind {
		case "NotDirectory", "AlreadyExists":
			return SandboxFilesystemNotADirectoryError{Exception: p.Message + ": " + remotePath}
		case "IsDirectory":
			return SandboxFilesystemIsADirectoryError{Exception: p.Message + ": " + remotePath}
		case "PermissionDenied":
			return SandboxFilesystemPermissionError{Exception: p.Message + ": " + remotePath}
		default:
			return SandboxFilesystemError{Exception: p.Message}
		}
	}
	if text := strings.TrimSpace(string(stderr)); text != "" {
		log.DebugContext(ctx, "Unstructured modal-sandbox-fs-tools stderr", "stderr", text)
	}
	return SandboxFilesystemError{Exception: fmt.Sprintf("Operation on '%s' failed with exit code %d", remotePath, returnCode)}
}

func makeWriteFileCommand(remotePath string) string {
	b, _ := json.Marshal(map[string]any{"WriteFile": map[string]any{"path": remotePath}})
	return string(b)
}
