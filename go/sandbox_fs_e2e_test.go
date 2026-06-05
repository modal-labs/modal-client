//go:build integration

package modal

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

// ---------------------------------------------------------------------------
// Sandbox fixture
// ---------------------------------------------------------------------------

func newTestSandbox(t *testing.T) *Sandbox {
	t.Helper()
	ctx := context.Background()

	tc, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &AppFromNameParams{CreateIfMissing: true})
	if err != nil {
		t.Fatalf("Apps.FromName: %v", err)
	}

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	if err != nil {
		t.Fatalf("Sandboxes.Create: %v", err)
	}

	t.Cleanup(func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		sb.Terminate(shutdownCtx, nil) //nolint
	})

	return sb
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func writeRemoteFile(ctx context.Context, sb *Sandbox, path string, data []byte) error {
	cmd := `mkdir -p "$(dirname '` + path + `')" && cat > '` + path + `'`
	cp, err := sb.Exec(ctx, []string{"sh", "-c", cmd}, nil)
	if err != nil {
		return err
	}
	defer cp.Stdout.Close()
	defer cp.Stderr.Close()
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

func readRemoteFile(ctx context.Context, sb *Sandbox, path string) ([]byte, error) {
	cp, err := sb.Exec(ctx, []string{"cat", path}, nil)
	if err != nil {
		return nil, err
	}
	defer cp.Stderr.Close()
	rc, err := cp.Wait(ctx, nil)
	stdout, _ := io.ReadAll(cp.Stdout)
	if err != nil {
		return nil, err
	}
	if rc != 0 {
		return nil, fmt.Errorf("readRemoteFile: %s exited %d", path, rc)
	}
	return stdout, nil
}

func mkdirRemote(ctx context.Context, sb *Sandbox, path string) error {
	return execRc(ctx, sb, []string{"mkdir", "-p", path})
}

func pathExists(ctx context.Context, sb *Sandbox, path string) (bool, error) {
	cp, err := sb.Exec(ctx, []string{"test", "-e", path}, nil)
	if err != nil {
		return false, err
	}
	defer cp.Stdout.Close()
	defer cp.Stderr.Close()
	rc, err := cp.Wait(ctx, nil)
	return rc == 0, err
}

func isDirRemote(ctx context.Context, sb *Sandbox, path string) (bool, error) {
	cp, err := sb.Exec(ctx, []string{"test", "-d", path}, nil)
	if err != nil {
		return false, err
	}
	defer cp.Stdout.Close()
	defer cp.Stderr.Close()
	rc, err := cp.Wait(ctx, nil)
	return rc == 0, err
}

func createSparseFile(ctx context.Context, sb *Sandbox, path string, sizeBytes int64) error {
	return execRc(ctx, sb, []string{"truncate", "-s", strconv.FormatInt(sizeBytes, 10), path})
}

func symlinkRemote(ctx context.Context, sb *Sandbox, target, linkPath string) error {
	return execRc(ctx, sb, []string{"ln", "-s", target, linkPath})
}

type remoteStatResult struct {
	permissions string
	owner       string
	group       string
	mtime       float64
	mode        int64
}

// Runs `stat -c '%a %U %G %Y %f' path` in the Sandbox for ground-truth metadata.
func statRemoteFile(ctx context.Context, sb *Sandbox, path string) (remoteStatResult, error) {
	cp, err := sb.Exec(ctx, []string{"stat", "-c", "%a %U %G %Y %f", path}, nil)
	if err != nil {
		return remoteStatResult{}, err
	}
	defer cp.Stderr.Close()
	rc, err := cp.Wait(ctx, nil)
	stdout, _ := io.ReadAll(cp.Stdout)
	if err != nil {
		return remoteStatResult{}, err
	}
	if rc != 0 {
		return remoteStatResult{}, fmt.Errorf("statRemoteFile: %s exited %d", path, rc)
	}
	parts := strings.Fields(strings.TrimSpace(string(stdout)))
	if len(parts) < 5 {
		return remoteStatResult{}, fmt.Errorf("statRemoteFile: unexpected output: %q", string(stdout))
	}
	perms := parts[0]
	for len(perms) < 4 {
		perms = "0" + perms
	}
	mtime, err := strconv.ParseFloat(parts[3], 64)
	if err != nil {
		return remoteStatResult{}, fmt.Errorf("statRemoteFile: parse mtime: %w", err)
	}
	mode, err := strconv.ParseInt(parts[4], 16, 64)
	if err != nil {
		return remoteStatResult{}, fmt.Errorf("statRemoteFile: parse mode: %w", err)
	}
	return remoteStatResult{permissions: perms, owner: parts[1], group: parts[2], mtime: mtime, mode: mode}, nil
}

// Runs args in the Sandbox and returns the exit code.
func execRc(ctx context.Context, sb *Sandbox, args []string) error {
	cp, err := sb.Exec(ctx, args, nil)
	if err != nil {
		return err
	}
	defer cp.Stdout.Close()
	defer cp.Stderr.Close()
	rc, err := cp.Wait(ctx, nil)
	if err != nil {
		return err
	}
	if rc != 0 {
		return fmt.Errorf("%v exited %d", args, rc)
	}
	return nil
}

// Generate deterministic pseudorandom bytes.
func randomBytes(size int, seed uint32) []byte {
	state := seed
	buf := make([]byte, size)
	for i := range buf {
		state = state*1664525 + 1013904223
		buf[i] = byte(state & 0xff)
	}
	return buf
}

// ---------------------------------------------------------------------------
// round-trips
// ---------------------------------------------------------------------------

func TestSandboxFsE2eWriteBytesReadBytesRoundTrip(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	payload := randomBytes(4096, 50)

	g.Expect(sb.Filesystem.WriteBytes(ctx, payload, "/tmp/e2e-rt-bytes.bin", nil)).To(gomega.Succeed())
	result, err := sb.Filesystem.ReadBytes(ctx, "/tmp/e2e-rt-bytes.bin", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(payload))
}

func TestSandboxFsE2eWriteTextReadTextRoundTrip(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	text := "round-trip text\nwith unicode: ☃🎉\n"

	g.Expect(sb.Filesystem.WriteText(ctx, text, "/tmp/e2e-rt-text.txt", nil)).To(gomega.Succeed())
	result, err := sb.Filesystem.ReadText(ctx, "/tmp/e2e-rt-text.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(text))
}

func TestSandboxFsE2eCopyFromLocalThenCopyToLocalRoundTrip(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	payload := randomBytes(8192, 60)
	localUp := filepath.Join(dir, "upload.bin")
	g.Expect(os.WriteFile(localUp, payload, 0o644)).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyFromLocal(ctx, localUp, "/tmp/e2e-copy-round-trip.bin", nil)).To(gomega.Succeed())
	localDown := filepath.Join(dir, "download.bin")
	g.Expect(sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-copy-round-trip.bin", localDown, nil)).To(gomega.Succeed())
	downloaded, err := os.ReadFile(localDown)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(downloaded).To(gomega.Equal(payload))
}

// ---------------------------------------------------------------------------
// copy_from_local
// ---------------------------------------------------------------------------

func TestSandboxFsE2eCopyFromLocalCopiesTextFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	g.Expect(os.WriteFile(filepath.Join(dir, "text.txt"), []byte("text content"), 0o644)).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyFromLocal(ctx, filepath.Join(dir, "text.txt"), "/tmp/e2e-cfl-text.txt", nil)).To(gomega.Succeed())
	data, err := readRemoteFile(ctx, sb, "/tmp/e2e-cfl-text.txt")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(data)).To(gomega.Equal("text content"))
}

func TestSandboxFsE2eCopyFromLocalCopiesEmptyFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	g.Expect(os.WriteFile(filepath.Join(dir, "empty.bin"), []byte{}, 0o644)).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyFromLocal(ctx, filepath.Join(dir, "empty.bin"), "/tmp/e2e-cfl-empty.bin", nil)).To(gomega.Succeed())
	data, err := readRemoteFile(ctx, sb, "/tmp/e2e-cfl-empty.bin")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(data).To(gomega.Equal([]byte{}))
}

func TestSandboxFsE2eCopyFromLocalHandlesLargeFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	payload := randomBytes(taskCommandRouterMaxBufferSize+1024, 70)
	g.Expect(os.WriteFile(filepath.Join(dir, "large.bin"), payload, 0o644)).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyFromLocal(ctx, filepath.Join(dir, "large.bin"), "/tmp/e2e-cfl-large.bin", nil)).To(gomega.Succeed())
	result, err := readRemoteFile(ctx, sb, "/tmp/e2e-cfl-large.bin")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(payload))
}

func TestSandboxFsE2eCopyFromLocalErrorsWhenLocalPathMissing(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()

	err := sb.Filesystem.CopyFromLocal(ctx, filepath.Join(dir, "missing.bin"), "/tmp/e2e-cfl-unused.bin", nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(os.IsNotExist(err)).To(gomega.BeTrue())
}

func TestSandboxFsE2eCopyFromLocalErrorsWhenLocalPathIsDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	g.Expect(os.MkdirAll(filepath.Join(dir, "src-dir"), 0o755)).To(gomega.Succeed())

	err := sb.Filesystem.CopyFromLocal(ctx, filepath.Join(dir, "src-dir"), "/tmp/e2e-cfl-unused2.bin", nil)
	g.Expect(err).To(gomega.HaveOccurred())
}

// ---------------------------------------------------------------------------
// copy_to_local
// ---------------------------------------------------------------------------

func TestSandboxFsE2eCopyToLocalCreatesParentDirectoriesIfNeeded(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	payload := randomBytes(2048, 2)
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ctl-parent.bin", payload)).To(gomega.Succeed())

	localPath := filepath.Join(dir, "deep", "nested", "path", "copied.bin")
	g.Expect(sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-ctl-parent.bin", localPath, nil)).To(gomega.Succeed())
	got, err := os.ReadFile(localPath)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal(payload))
}

func TestSandboxFsE2eCopyToLocalCopiesEmptyFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ctl-empty.bin", []byte{})).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-ctl-empty.bin", filepath.Join(dir, "empty.bin"), nil)).To(gomega.Succeed())
	got, err := os.ReadFile(filepath.Join(dir, "empty.bin"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal([]byte{}))
}

func TestSandboxFsE2eCopyToLocalOverwritesExistingLocalFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	payload := randomBytes(4096, 5)
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ctl-overwrite.bin", payload)).To(gomega.Succeed())
	g.Expect(os.WriteFile(filepath.Join(dir, "overwrite.bin"), []byte("old-data"), 0o644)).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-ctl-overwrite.bin", filepath.Join(dir, "overwrite.bin"), nil)).To(gomega.Succeed())
	got, err := os.ReadFile(filepath.Join(dir, "overwrite.bin"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal(payload))
}

func TestSandboxFsE2eCopyToLocalPreservesExistingFileOnRemoteError(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	existing := filepath.Join(dir, "existing.bin")
	g.Expect(os.WriteFile(existing, []byte("stable-content"), 0o644)).To(gomega.Succeed())

	err := sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-copy-to-local-missing.bin", existing, nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
	got, readErr := os.ReadFile(existing)
	g.Expect(readErr).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(got)).To(gomega.Equal("stable-content"))
}

func TestSandboxFsE2eCopyToLocalErrorsWhenLocalPathIsDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ctl-dir-source.bin", randomBytes(512, 6))).To(gomega.Succeed())
	localDir := filepath.Join(dir, "local-dir")
	g.Expect(os.MkdirAll(localDir, 0o755)).To(gomega.Succeed())

	g.Expect(sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-ctl-dir-source.bin", localDir, nil)).To(gomega.HaveOccurred())
	entries, err := os.ReadDir(dir)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for _, e := range entries {
		g.Expect(e.Name()).NotTo(gomega.HavePrefix(".modal-sandbox-fs-tmp-"))
	}
}

func TestSandboxFsE2eCopyToLocalErrorsWhenRemotePathIsDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ctl-remote-dir")).To(gomega.Succeed())

	err := sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-ctl-remote-dir", filepath.Join(dir, "unused.bin"), nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemIsADirectoryError{}))
}

func TestSandboxFsE2eCopyToLocalErrorsWhenFileTooLarge(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	dir := t.TempDir()
	localPath := filepath.Join(dir, "too-large-out.bin")
	g.Expect(createSparseFile(ctx, sb, "/tmp/e2e-copy-too-large.bin", 6*1024*1024*1024)).To(gomega.Succeed())

	err := sb.Filesystem.CopyToLocal(ctx, "/tmp/e2e-copy-too-large.bin", localPath, nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemFileTooLargeError{}))
	_, statErr := os.Stat(localPath)
	g.Expect(os.IsNotExist(statErr)).To(gomega.BeTrue())
}

// ---------------------------------------------------------------------------
// list_files
// ---------------------------------------------------------------------------

func TestSandboxFsE2eListFilesReturnsFilesAndDirectories(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-list-files-basic")).To(gomega.Succeed())
	content := []byte("hello list_files")
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-list-files-basic/file.txt", content)).To(gomega.Succeed())
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-list-files-basic/subdir")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-list-files-basic", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	names := make(map[string]FileInfo)
	for _, e := range entries {
		names[e.Name] = e
	}
	g.Expect(names).To(gomega.HaveKey("file.txt"))
	g.Expect(names).To(gomega.HaveKey("subdir"))
	g.Expect(names["file.txt"].Type).To(gomega.Equal(FileTypeFile))
	g.Expect(names["file.txt"].Size).To(gomega.Equal(int64(len(content))))
	g.Expect(names["subdir"].Type).To(gomega.Equal(FileTypeDirectory))
}

func TestSandboxFsE2eListFilesIsNotRecursive(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-nonrecursive")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-nonrecursive/top.txt", []byte("top"))).To(gomega.Succeed())
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-nonrecursive/child")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-nonrecursive/child/nested.txt", []byte("nested"))).To(gomega.Succeed())
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-nonrecursive/child/grandchild")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-nonrecursive", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	names := make([]string, len(entries))
	for i, e := range entries {
		names[i] = e.Name
	}
	g.Expect(names).To(gomega.ConsistOf("top.txt", "child"))
}

func TestSandboxFsE2eListFilesFileInfoHasCorrectMetadata(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	content := []byte("field check")
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-list-files-fields")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-list-files-fields/check.txt", content)).To(gomega.Succeed())
	expected, err := statRemoteFile(ctx, sb, "/tmp/e2e-list-files-fields/check.txt")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-list-files-fields", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	var entry FileInfo
	for _, e := range entries {
		if e.Name == "check.txt" {
			entry = e
			break
		}
	}
	g.Expect(entry.Type).To(gomega.Equal(FileTypeFile))
	g.Expect(entry.Size).To(gomega.Equal(int64(len(content))))
	g.Expect(entry.Permissions).To(gomega.Equal(expected.permissions))
	g.Expect(int64(entry.Mode)).To(gomega.Equal(expected.mode))
	g.Expect(entry.Owner).To(gomega.Equal(expected.owner))
	g.Expect(entry.Group).To(gomega.Equal(expected.group))
	g.Expect(math.Abs(entry.ModifiedTime - expected.mtime)).To(gomega.BeNumerically("<=", 10))
	g.Expect(entry.SymlinkTarget).To(gomega.BeNil())
}

func TestSandboxFsE2eListFilesReturnsEmptyListForEmptyDir(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-empty")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-empty", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(entries).To(gomega.BeEmpty())
}

func TestSandboxFsE2eListFilesEntriesSortedByName(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-sorted")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-sorted/zebra.txt", []byte("z"))).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-sorted/alpha.txt", []byte("a"))).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-sorted/middle.txt", []byte("m"))).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-sorted", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	names := make([]string, len(entries))
	for i, e := range entries {
		names[i] = e.Name
	}
	g.Expect(names).To(gomega.Equal([]string{"alpha.txt", "middle.txt", "zebra.txt"}))
}

func TestSandboxFsE2eListFilesErrorsWhenPathDoesNotExist(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	_, err := sb.Filesystem.ListFiles(context.Background(), "/tmp/e2e-ls-nonexistent", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
}

func TestSandboxFsE2eListFilesErrorsWhenPathIsAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-isfile.txt", []byte("not a dir"))).To(gomega.Succeed())

	_, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-isfile.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotADirectoryError{}))
}

func TestSandboxFsE2eListFilesErrorsWhenPathComponentIsAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-blocker.txt", []byte("file"))).To(gomega.Succeed())

	_, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-blocker.txt/subdir", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotADirectoryError{}))
}

func TestSandboxFsE2eListFilesSymlinkReportedAsSymlinkWithTarget(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-symlink")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-symlink/target.txt", []byte("hi"))).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-ls-symlink/target.txt", "/tmp/e2e-ls-symlink/link.txt")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-symlink", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	var link *FileInfo
	for i := range entries {
		if entries[i].Name == "link.txt" {
			link = &entries[i]
		}
	}
	g.Expect(link).NotTo(gomega.BeNil())
	g.Expect(link.Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(link.SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*link.SymlinkTarget).To(gomega.Equal("/tmp/e2e-ls-symlink/target.txt"))
}

func TestSandboxFsE2eListFilesDoesNotShowSymlinkTargetForNonsymlinkedFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-no-symlink-file")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-no-symlink-file/file.txt", []byte("hello"))).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-no-symlink-file", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(entries).To(gomega.HaveLen(1))
	g.Expect(entries[0].SymlinkTarget).To(gomega.BeNil())
}

func TestSandboxFsE2eListFilesDoesNotShowSymlinkTargetForNonsymlinkedDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-no-symlink-dir")).To(gomega.Succeed())
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-no-symlink-dir/subdir")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-no-symlink-dir", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(entries).To(gomega.HaveLen(1))
	g.Expect(entries[0].SymlinkTarget).To(gomega.BeNil())
}

func TestSandboxFsE2eListFilesDanglingSymlinkReportedAsSymlink(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-dangling")).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-ls-dangling/nonexistent", "/tmp/e2e-ls-dangling/dangling")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-dangling", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(entries).To(gomega.HaveLen(1))
	g.Expect(entries[0].Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(entries[0].SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*entries[0].SymlinkTarget).To(gomega.Equal("/tmp/e2e-ls-dangling/nonexistent"))
}

func TestSandboxFsE2eListFilesFollowsSymlinkIfPathIsDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-follow-target")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-ls-follow-target/file.txt", []byte("hello"))).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-ls-follow-target", "/tmp/e2e-ls-follow-link")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-follow-link", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(entries).To(gomega.HaveLen(1))
	g.Expect(entries[0].Name).To(gomega.Equal("file.txt"))
	g.Expect(entries[0].Type).To(gomega.Equal(FileTypeFile))
	g.Expect(entries[0].Path).To(gomega.ContainSubstring("e2e-ls-follow-link"))
}

func TestSandboxFsE2eListFilesSymlinkToDirectoryReportedAsSymlink(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-dirlink")).To(gomega.Succeed())
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-ls-dirlink-target")).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-ls-dirlink-target", "/tmp/e2e-ls-dirlink/link-to-dir")).To(gomega.Succeed())

	entries, err := sb.Filesystem.ListFiles(ctx, "/tmp/e2e-ls-dirlink", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	var link *FileInfo
	for i := range entries {
		if entries[i].Name == "link-to-dir" {
			link = &entries[i]
		}
	}
	g.Expect(link).NotTo(gomega.BeNil())
	g.Expect(link.Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(link.SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*link.SymlinkTarget).To(gomega.Equal("/tmp/e2e-ls-dirlink-target"))
}

// ---------------------------------------------------------------------------
// make_directory
// ---------------------------------------------------------------------------

func TestSandboxFsE2eMakeDirectoryCreatesNestedDirectories(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-make-dir-a/b/c", nil)).To(gomega.Succeed())
	ok, err := isDirRemote(ctx, sb, "/tmp/e2e-make-dir-a/b/c")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(ok).To(gomega.BeTrue())
}

func TestSandboxFsE2eMakeDirectoryNoParentsCreatesDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	f := false
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-make-dir-parent")).To(gomega.Succeed())
	g.Expect(sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-make-dir-parent/new-subdir",
		&SandboxFilesystemMakeDirectoryParams{CreateParents: &f})).To(gomega.Succeed())
	ok, err := isDirRemote(ctx, sb, "/tmp/e2e-make-dir-parent/new-subdir")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(ok).To(gomega.BeTrue())
}

func TestSandboxFsE2eMakeDirectoryIsIdempotentWhenAlreadyExists(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-mkdir-idem")).To(gomega.Succeed())
	g.Expect(sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-mkdir-idem", nil)).To(gomega.Succeed())
	ok, err := isDirRemote(ctx, sb, "/tmp/e2e-mkdir-idem")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(ok).To(gomega.BeTrue())
}

func TestSandboxFsE2eMakeDirectoryNoParentsErrorsWhenAlreadyExists(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	f := false
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-make-dir-existing")).To(gomega.Succeed())
	err := sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-make-dir-existing",
		&SandboxFilesystemMakeDirectoryParams{CreateParents: &f})
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemPathAlreadyExistsError{}))
}

func TestSandboxFsE2eMakeDirectoryNoParentsErrorsWhenParentMissing(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	f := false
	err := sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-mkdir-missing-parent/child",
		&SandboxFilesystemMakeDirectoryParams{CreateParents: &f})
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
}

func TestSandboxFsE2eMakeDirectoryNoParentsErrorsWhenTargetIsAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	f := false
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-mkdir-target-file", []byte("file"))).To(gomega.Succeed())
	err := sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-mkdir-target-file",
		&SandboxFilesystemMakeDirectoryParams{CreateParents: &f})
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemPathAlreadyExistsError{}))
}

func TestSandboxFsE2eMakeDirectoryErrorsWhenTargetIsAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-mkdir-target-file-parents", []byte("file"))).To(gomega.Succeed())
	err := sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-mkdir-target-file-parents", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemPathAlreadyExistsError{}))
}

func TestSandboxFsE2eMakeDirectoryErrorsWhenAncestorIsAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-mkdir-blocker", []byte("file"))).To(gomega.Succeed())
	err := sb.Filesystem.MakeDirectory(ctx, "/tmp/e2e-mkdir-blocker/child", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotADirectoryError{}))
}

// ---------------------------------------------------------------------------
// read_bytes
// ---------------------------------------------------------------------------

func TestSandboxFsE2eReadBytesReturnsExpectedBytes(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	payload := []byte{0x00, 0x01, 0x02, 0x62, 0x69, 0x6e, 0xff}
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-read-bytes.bin", payload)).To(gomega.Succeed())

	result, err := sb.Filesystem.ReadBytes(ctx, "/tmp/e2e-read-bytes.bin", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(payload))
}

func TestSandboxFsE2eReadBytesReturnsEmptyBytesForEmptyFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-read-bytes-empty.bin", []byte{})).To(gomega.Succeed())

	result, err := sb.Filesystem.ReadBytes(ctx, "/tmp/e2e-read-bytes-empty.bin", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal([]byte{}))
}

func TestSandboxFsE2eReadBytesErrorsWhenRemotePathMissing(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	_, err := sb.Filesystem.ReadBytes(context.Background(), "/tmp/e2e-read-bytes-missing.bin", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
}

func TestSandboxFsE2eReadBytesErrorsWhenRemotePathIsDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-read-bytes-dir")).To(gomega.Succeed())

	_, err := sb.Filesystem.ReadBytes(ctx, "/tmp/e2e-read-bytes-dir", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemIsADirectoryError{}))
}

func TestSandboxFsE2eReadBytesErrorsWhenFileTooLarge(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(createSparseFile(ctx, sb, "/tmp/e2e-read-bytes-large.bin", 6*1024*1024*1024)).To(gomega.Succeed())

	_, err := sb.Filesystem.ReadBytes(ctx, "/tmp/e2e-read-bytes-large.bin", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemFileTooLargeError{}))
}

// ---------------------------------------------------------------------------
// read_text
// ---------------------------------------------------------------------------

func TestSandboxFsE2eReadTextReturnsExpectedText(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	text := "hello from read_text\nsnowman: ☃\n"
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-read-text.txt", []byte(text))).To(gomega.Succeed())

	result, err := sb.Filesystem.ReadText(ctx, "/tmp/e2e-read-text.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(text))
}

func TestSandboxFsE2eReadTextReturnsEmptyStringForEmptyFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-read-text-empty.txt", []byte{})).To(gomega.Succeed())

	result, err := sb.Filesystem.ReadText(ctx, "/tmp/e2e-read-text-empty.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(""))
}

func TestSandboxFsE2eReadTextErrorsWhenRemotePathMissing(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	_, err := sb.Filesystem.ReadText(context.Background(), "/tmp/e2e-read-text-missing.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
}

func TestSandboxFsE2eReadTextErrorsWhenRemotePathIsDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-read-text-dir")).To(gomega.Succeed())

	_, err := sb.Filesystem.ReadText(ctx, "/tmp/e2e-read-text-dir", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemIsADirectoryError{}))
}

func TestSandboxFsE2eReadTextErrorsWhenFileTooLarge(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(createSparseFile(ctx, sb, "/tmp/e2e-read-text-large.txt", 6*1024*1024*1024)).To(gomega.Succeed())

	_, err := sb.Filesystem.ReadText(ctx, "/tmp/e2e-read-text-large.txt", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemFileTooLargeError{}))
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

func TestSandboxFsE2eRemoveRemovesAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-remove-file.bin", []byte("data"))).To(gomega.Succeed())

	g.Expect(sb.Filesystem.Remove(ctx, "/tmp/e2e-remove-file.bin", nil)).To(gomega.Succeed())
	ok, err := pathExists(ctx, sb, "/tmp/e2e-remove-file.bin")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(ok).To(gomega.BeFalse())
}

func TestSandboxFsE2eRemoveRemovesEmptyDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-rm-emptydir")).To(gomega.Succeed())

	g.Expect(sb.Filesystem.Remove(ctx, "/tmp/e2e-rm-emptydir", nil)).To(gomega.Succeed())
	ok, err := pathExists(ctx, sb, "/tmp/e2e-rm-emptydir")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(ok).To(gomega.BeFalse())
}

func TestSandboxFsE2eRemoveRecursiveRemovesDirectoryTree(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-remove-tree/a/b")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-remove-tree/a/file.txt", []byte("data"))).To(gomega.Succeed())

	g.Expect(sb.Filesystem.Remove(ctx, "/tmp/e2e-remove-tree", &SandboxFilesystemRemoveParams{Recursive: true})).To(gomega.Succeed())
	ok, err := pathExists(ctx, sb, "/tmp/e2e-remove-tree")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(ok).To(gomega.BeFalse())
}

func TestSandboxFsE2eRemoveErrorsWhenMissing(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	err := sb.Filesystem.Remove(context.Background(), "/tmp/e2e-rm-missing", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
}

func TestSandboxFsE2eRemoveErrorsWhenTargetIsNonemptyDirectoryAndNotRecursive(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-rm-nonempty")).To(gomega.Succeed())
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-rm-nonempty/file.txt", []byte("hello"))).To(gomega.Succeed())

	err := sb.Filesystem.Remove(ctx, "/tmp/e2e-rm-nonempty", &SandboxFilesystemRemoveParams{Recursive: false})
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemDirectoryNotEmptyError{}))
}

func TestSandboxFsE2eRemoveRemovesSymlinkWithoutFollowingIt(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-rm-symlink-target.txt", []byte("original"))).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-rm-symlink-target.txt", "/tmp/e2e-rm-symlink-link.txt")).To(gomega.Succeed())

	g.Expect(sb.Filesystem.Remove(ctx, "/tmp/e2e-rm-symlink-link.txt", nil)).To(gomega.Succeed())
	linkExists, err := pathExists(ctx, sb, "/tmp/e2e-rm-symlink-link.txt")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(linkExists).To(gomega.BeFalse())
	targetExists, err := pathExists(ctx, sb, "/tmp/e2e-rm-symlink-target.txt")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(targetExists).To(gomega.BeTrue())
}

// ---------------------------------------------------------------------------
// stat
// ---------------------------------------------------------------------------

func TestSandboxFsE2eStatReturnsMetadataForFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	content := []byte("hello stat")
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-stat-file.txt", content)).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-file.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Name).To(gomega.Equal("e2e-stat-file.txt"))
	g.Expect(info.Path).To(gomega.Equal("/tmp/e2e-stat-file.txt"))
	g.Expect(info.Type).To(gomega.Equal(FileTypeFile))
	g.Expect(info.Size).To(gomega.Equal(int64(len(content))))
	g.Expect(info.Permissions).To(gomega.MatchRegexp(`^\d{4}$`))
	g.Expect(info.Mode).To(gomega.BeNumerically(">", 0))
	g.Expect(info.ModifiedTime).To(gomega.BeNumerically(">", 0))
	g.Expect(info.SymlinkTarget).To(gomega.BeNil())
}

func TestSandboxFsE2eStatReturnsMetadataForDirectory(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-stat-dir")).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-dir", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Name).To(gomega.Equal("e2e-stat-dir"))
	g.Expect(info.Type).To(gomega.Equal(FileTypeDirectory))
	g.Expect(info.SymlinkTarget).To(gomega.BeNil())
}

func TestSandboxFsE2eStatReturnsMetadataForEmptyFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-stat-empty.txt", []byte{})).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-empty.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Type).To(gomega.Equal(FileTypeFile))
	g.Expect(info.Size).To(gomega.Equal(int64(0)))
}

func TestSandboxFsE2eStatExactFieldsMatchShellStat(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	content := []byte("field check")
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-stat-fields.txt", content)).To(gomega.Succeed())
	expected, err := statRemoteFile(ctx, sb, "/tmp/e2e-stat-fields.txt")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-fields.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Permissions).To(gomega.Equal(expected.permissions))
	g.Expect(int64(info.Mode)).To(gomega.Equal(expected.mode))
	g.Expect(info.Owner).To(gomega.Equal(expected.owner))
	g.Expect(info.Group).To(gomega.Equal(expected.group))
	g.Expect(math.Abs(info.ModifiedTime - expected.mtime)).To(gomega.BeNumerically("<=", 10))
	g.Expect(info.SymlinkTarget).To(gomega.BeNil())
}

func TestSandboxFsE2eStatSymlinkToFileReportedAsSymlink(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-stat-lnk-target.txt", []byte("hi"))).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-stat-lnk-target.txt", "/tmp/e2e-stat-lnk.txt")).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-lnk.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(info.SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*info.SymlinkTarget).To(gomega.Equal("/tmp/e2e-stat-lnk-target.txt"))
}

func TestSandboxFsE2eStatSymlinkToDirectoryReportedAsSymlink(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(mkdirRemote(ctx, sb, "/tmp/e2e-stat-dir-link-target")).To(gomega.Succeed())
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-stat-dir-link-target", "/tmp/e2e-stat-dir-link")).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-dir-link", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(info.SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*info.SymlinkTarget).To(gomega.Equal("/tmp/e2e-stat-dir-link-target"))
}

func TestSandboxFsE2eStatDanglingSymlinkReportedAsSymlink(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(symlinkRemote(ctx, sb, "/tmp/e2e-stat-dangling-target", "/tmp/e2e-stat-dangling.txt")).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-dangling.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(info.SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*info.SymlinkTarget).To(gomega.Equal("/tmp/e2e-stat-dangling-target"))
}

func TestSandboxFsE2eStatRelativeSymlinkTargetPreserved(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(symlinkRemote(ctx, sb, "target.txt", "/tmp/e2e-stat-rel-link.txt")).To(gomega.Succeed())

	info, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-rel-link.txt", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(info.Type).To(gomega.Equal(FileTypeSymlink))
	g.Expect(info.SymlinkTarget).NotTo(gomega.BeNil())
	g.Expect(*info.SymlinkTarget).To(gomega.Equal("target.txt"))
}

func TestSandboxFsE2eStatErrorsWhenPathDoesNotExist(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	_, err := sb.Filesystem.Stat(context.Background(), "/tmp/e2e-stat-nonexistent", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotFoundError{}))
}

func TestSandboxFsE2eStatErrorsWhenAncestorIsAFile(t *testing.T) {
	sb := newTestSandbox(t)
	g := gomega.NewWithT(t)
	ctx := context.Background()
	g.Expect(writeRemoteFile(ctx, sb, "/tmp/e2e-stat-blocker", []byte("I am a file"))).To(gomega.Succeed())

	_, err := sb.Filesystem.Stat(ctx, "/tmp/e2e-stat-blocker/child", nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(SandboxFilesystemNotADirectoryError{}))
}
