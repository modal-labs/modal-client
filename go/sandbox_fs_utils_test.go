package modal

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"testing"

	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ---------------------------------------------------------------------------
// tryParseErrorPayload
// ---------------------------------------------------------------------------

func TestTryParseErrorPayloadReturnsPayloadForValidJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	stderr := []byte(`{"error_kind":"NotFound","message":"file not found"}`)

	p := tryParseErrorPayload(stderr)

	g.Expect(p).NotTo(gomega.BeNil())
	g.Expect(p.ErrorKind).To(gomega.Equal("NotFound"))
	g.Expect(p.Message).To(gomega.Equal("file not found"))
	g.Expect(p.Detail).To(gomega.Equal(""))
}

func TestTryParseErrorPayloadReturnsNilForEmptyBytes(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte{})).To(gomega.BeNil())
	g.Expect(tryParseErrorPayload([]byte("   "))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForNonJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte("not json at all"))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForNonObjectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte(`[1,2,3]`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForMissingErrorKind(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte(`{"message":"oops"}`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForNonStringErrorKind(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// json.Unmarshal rejects a number in a string field, so the whole parse fails.
	g.Expect(tryParseErrorPayload([]byte(`{"error_kind":42,"message":"oops"}`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForMissingMessage(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte(`{"error_kind":"NotFound"}`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForNonStringMessage(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// json.Unmarshal rejects a number in a string field.
	g.Expect(tryParseErrorPayload([]byte(`{"error_kind":"NotFound","message":123}`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadReturnsNilForBlankMessage(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte(`{"error_kind":"NotFound","message":"  "}`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadIncludesDetailWhenPresent(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	stderr := []byte(`{"error_kind":"Io","message":"I/O error","detail":"No such file or directory (os error 2)"}`)

	p := tryParseErrorPayload(stderr)

	g.Expect(p).NotTo(gomega.BeNil())
	g.Expect(p.ErrorKind).To(gomega.Equal("Io"))
	g.Expect(p.Message).To(gomega.Equal("I/O error"))
	g.Expect(p.Detail).To(gomega.Equal("No such file or directory (os error 2)"))
}

// JS's tryParseErrorPayload coerces a non-string detail to ""; Go's json.Unmarshal
// is stricter and fails the entire parse when a string field receives a number.
func TestTryParseErrorPayloadReturnsNilForNonStringDetail(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	g.Expect(tryParseErrorPayload([]byte(`{"error_kind":"Io","message":"I/O error","detail":42}`))).To(gomega.BeNil())
}

func TestTryParseErrorPayloadUsesEmptyStringForMissingDetail(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	stderr := []byte(`{"error_kind":"Io","message":"I/O error"}`)

	p := tryParseErrorPayload(stderr)

	g.Expect(p).NotTo(gomega.BeNil())
	g.Expect(p.Detail).To(gomega.Equal(""))
}

// ---------------------------------------------------------------------------
// command builders
// ---------------------------------------------------------------------------

func TestMakeListFilesCommandProducesCorrectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var got map[string]any
	g.Expect(json.Unmarshal([]byte(makeListFilesCommand("/tmp/mydir")), &got)).To(gomega.Succeed())

	g.Expect(got).To(gomega.Equal(map[string]any{
		"ListFiles": map[string]any{"path": "/tmp/mydir"},
	}))
}

func TestMakeListFilesCommandHandlesPathsWithSpecialCharacters(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var got map[string]any
	g.Expect(json.Unmarshal([]byte(makeListFilesCommand("/tmp/my dir/with spaces")), &got)).To(gomega.Succeed())

	g.Expect(got).To(gomega.Equal(map[string]any{
		"ListFiles": map[string]any{"path": "/tmp/my dir/with spaces"},
	}))
}

func TestMakeStatCommandProducesCorrectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var got map[string]any
	g.Expect(json.Unmarshal([]byte(makeStatCommand("/tmp/file.txt")), &got)).To(gomega.Succeed())

	g.Expect(got).To(gomega.Equal(map[string]any{
		"Stat": map[string]any{"path": "/tmp/file.txt"},
	}))
}

func TestMakeStatCommandHandlesPathsWithSpecialCharacters(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var got map[string]any
	g.Expect(json.Unmarshal([]byte(makeStatCommand("/tmp/my dir/with spaces/file.txt")), &got)).To(gomega.Succeed())

	g.Expect(got).To(gomega.Equal(map[string]any{
		"Stat": map[string]any{"path": "/tmp/my dir/with spaces/file.txt"},
	}))
}

func TestMakeReadFileCommandProducesCorrectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var got map[string]any
	g.Expect(json.Unmarshal([]byte(makeReadFileCommand("/tmp/file.txt")), &got)).To(gomega.Succeed())

	g.Expect(got).To(gomega.Equal(map[string]any{
		"ReadFile": map[string]any{"path": "/tmp/file.txt"},
	}))
}

func TestMakeWriteFileCommandProducesCorrectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var got map[string]any
	g.Expect(json.Unmarshal([]byte(makeWriteFileCommand("/tmp/file.txt")), &got)).To(gomega.Succeed())

	g.Expect(got).To(gomega.Equal(map[string]any{
		"WriteFile": map[string]any{"path": "/tmp/file.txt"},
	}))
}

func TestMakeMakeDirectoryCommandProducesCorrectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var withParents, noParents map[string]any
	g.Expect(json.Unmarshal([]byte(makeMakeDirectoryCommand("/tmp/dir", true)), &withParents)).To(gomega.Succeed())
	g.Expect(json.Unmarshal([]byte(makeMakeDirectoryCommand("/tmp/dir", false)), &noParents)).To(gomega.Succeed())

	g.Expect(withParents).To(gomega.Equal(map[string]any{
		"MakeDirectory": map[string]any{"path": "/tmp/dir", "parents": true},
	}))
	g.Expect(noParents).To(gomega.Equal(map[string]any{
		"MakeDirectory": map[string]any{"path": "/tmp/dir", "parents": false},
	}))
}

func TestMakeRemoveCommandProducesCorrectJSON(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var nonRecursive, recursive map[string]any
	g.Expect(json.Unmarshal([]byte(makeRemoveCommand("/tmp/file.txt", false)), &nonRecursive)).To(gomega.Succeed())
	g.Expect(json.Unmarshal([]byte(makeRemoveCommand("/tmp/dir", true)), &recursive)).To(gomega.Succeed())

	g.Expect(nonRecursive).To(gomega.Equal(map[string]any{
		"Remove": map[string]any{"path": "/tmp/file.txt", "recursive": false},
	}))
	g.Expect(recursive).To(gomega.Equal(map[string]any{
		"Remove": map[string]any{"path": "/tmp/dir", "recursive": true},
	}))
}

// ---------------------------------------------------------------------------
// translateExecError
// ---------------------------------------------------------------------------

func TestTranslateExecErrorIncludesSupportErrorCode(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	err := translateExecError(
		context.Background(),
		slog.Default(),
		"readBytes",
		"/tmp/missing.txt",
		status.Error(codes.Internal, "Failed to start exec command (Error code: ABCD1234)"),
	)

	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemError{}))
	g.Expect(err.Error()).To(gomega.ContainSubstring("Error code: ABCD1234"))
}

func TestTranslateExecErrorReturnsSandboxUnavailableForNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	err := translateExecError(
		context.Background(),
		slog.Default(),
		"readBytes",
		"/tmp/file.txt",
		status.Error(codes.NotFound, "sandbox not found"),
	)

	g.Expect(err).To(gomega.BeAssignableToTypeOf(NotFoundError{}))
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox is unavailable"))
}

func TestTranslateExecErrorReturnsCancelledContextError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := translateExecError(
		ctx,
		slog.Default(),
		"CopyToLocal",
		"/tmp/file.txt",
		status.Error(codes.Canceled, "rpc error: code = Canceled desc = context canceled"),
	)

	g.Expect(err).To(gomega.MatchError(context.Canceled))
}

func TestTranslateExecErrorDoesNotClassifyNonGRPCErrorAsSandboxUnavailable(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// A plain OS error (e.g. local disk full) must not be misclassified as
	// "Sandbox is unavailable" just because status.Code returns codes.Unknown
	// for non-gRPC errors.
	osErr := fmt.Errorf("write /tmp/foo: no space left on device")
	err := translateExecError(
		context.Background(),
		slog.Default(),
		"CopyToLocal",
		"/tmp/file.txt",
		osErr,
	)

	g.Expect(err).NotTo(gomega.BeAssignableToTypeOf(NotFoundError{}))
}

func TestTranslateExecErrorReturnsUnexpectedErrorForGenericError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	err := translateExecError(
		context.Background(),
		slog.Default(),
		"readBytes",
		"/tmp/file.txt",
		status.Error(codes.Internal, "something broke"),
	)

	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemError{}))
	g.Expect(err.Error()).To(gomega.ContainSubstring("unexpected error"))
}
