package modal

import (
	"context"
	"log/slog"
	"testing"

	"github.com/onsi/gomega"
)

// This satisfies sandboxForFilesystem and panics if exec is ever called.
type panicSandbox struct{}

func (panicSandbox) execForFilesystem(context.Context, []string, *SandboxExecParams) (*ContainerProcess, error) {
	panic("exec must not be called when path validation fails")
}

func newValidationFilesystem() *SandboxFilesystem {
	return &SandboxFilesystem{sandbox: panicSandbox{}, logger: slog.Default()}
}

// ---------------------------------------------------------------------------
// Unit tests that throw before exec.
// ---------------------------------------------------------------------------

func TestSandboxFsCopyFromLocalErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	err := newValidationFilesystem().CopyFromLocal(context.Background(), "/any/local/path", "relative/path.bin", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsCopyToLocalErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	err := newValidationFilesystem().CopyToLocal(context.Background(), "relative/path.bin", "/any/local/path", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsListFilesErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	_, err := newValidationFilesystem().ListFiles(context.Background(), "relative/path", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsMakeDirectoryErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	err := newValidationFilesystem().MakeDirectory(context.Background(), "relative/path", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsReadBytesErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	_, err := newValidationFilesystem().ReadBytes(context.Background(), "relative/path.bin", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsReadTextErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	_, err := newValidationFilesystem().ReadText(context.Background(), "relative/path.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsRemoveErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	err := newValidationFilesystem().Remove(context.Background(), "relative/path.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsStatErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	_, err := newValidationFilesystem().Stat(context.Background(), "relative/path.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsWatchErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	_, err := newValidationFilesystem().Watch(context.Background(), "relative/path", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsWriteBytesErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	err := newValidationFilesystem().WriteBytes(context.Background(), []byte("data"), "relative/path.bin", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}

func TestSandboxFsWriteTextErrorsOnRelativeRemotePath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	err := newValidationFilesystem().WriteText(context.Background(), "data", "relative/path.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
}
