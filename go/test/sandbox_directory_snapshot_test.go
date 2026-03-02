package test

import (
	"io"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func TestSandboxMountDirectoryEmpty(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("debian:12-slim", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	mkdirProc, err := sb.Exec(ctx, []string{"mkdir", "-p", "/mnt/empty"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = mkdirProc.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = sb.MountImage(ctx, "/mnt/empty", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	dirCheck, err := sb.Exec(ctx, []string{"test", "-d", "/mnt/empty"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	exitCode, err := dirCheck.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxMountDirectoryWithImage(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	baseImage := tc.Images.FromRegistry("debian:12-slim", nil)

	sb1, err := tc.Sandboxes.Create(ctx, app, baseImage, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	echoProc, err := sb1.Exec(ctx, []string{
		"sh",
		"-c",
		"echo -n 'mounted content' > /tmp/test.txt",
	}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = echoProc.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	mountImage, err := sb1.SnapshotFilesystem(ctx, 55*time.Second)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mountImage.ImageID).To(gomega.MatchRegexp(`^im-`))

	_, err = sb1.Terminate(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb2, err := tc.Sandboxes.Create(ctx, app, baseImage, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb2)

	mkdirProc, err := sb2.Exec(ctx, []string{"mkdir", "-p", "/mnt/data"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = mkdirProc.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = sb2.MountImage(ctx, "/mnt/data", mountImage)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	catProc, err := sb2.Exec(ctx, []string{"cat", "/mnt/data/tmp/test.txt"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	output, err := io.ReadAll(catProc.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("mounted content"))
}

func TestSandboxSnapshotDirectory(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	baseImage := tc.Images.FromRegistry("debian:12-slim", nil)

	sb1, err := tc.Sandboxes.Create(ctx, app, baseImage, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	sb1FromID, err := tc.Sandboxes.FromID(ctx, sb1.SandboxID)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb1FromID)

	mkdirProc, err := sb1.Exec(ctx, []string{"mkdir", "-p", "/mnt/data"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = mkdirProc.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = sb1.MountImage(ctx, "/mnt/data", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	echoProc, err := sb1.Exec(ctx, []string{
		"sh",
		"-c",
		"echo -n 'snapshot test content' > /mnt/data/snapshot.txt",
	}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = echoProc.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	snapshotImage, err := sb1.SnapshotDirectory(ctx, "/mnt/data")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(snapshotImage.ImageID).To(gomega.MatchRegexp(`^im-`))

	_, err = sb1.Terminate(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb2, err := tc.Sandboxes.Create(ctx, app, baseImage, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb2)

	mkdirProc2, err := sb2.Exec(ctx, []string{"mkdir", "-p", "/mnt/data"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = mkdirProc2.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = sb2.MountImage(ctx, "/mnt/data", snapshotImage)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	catProc, err := sb2.Exec(ctx, []string{"cat", "/mnt/data/snapshot.txt"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	output, err := io.ReadAll(catProc.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("snapshot test content"))
}

func TestSandboxMountDirectoryWithUnbuiltImageThrows(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	baseImage := tc.Images.FromRegistry("debian:12-slim", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, baseImage, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	mkdirProc, err := sb.Exec(ctx, []string{"mkdir", "-p", "/mnt/data"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = mkdirProc.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	unbuiltImage := tc.Images.FromRegistry("alpine:3.21", nil)
	g.Expect(unbuiltImage.ImageID).To(gomega.Equal(""))

	err = sb.MountImage(ctx, "/mnt/data", unbuiltImage)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Image must be built before mounting"))
}
