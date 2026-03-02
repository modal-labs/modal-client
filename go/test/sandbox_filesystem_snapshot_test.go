package test

import (
	"io"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func TestSnapshotFilesystem(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	sbFromID, err := tc.Sandboxes.FromID(ctx, sb.SandboxID)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbFromID)

	writeFile, err := sb.Exec(ctx, []string{"sh", "-c", "echo -n 'test content' > /tmp/test.txt"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = writeFile.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	mkDir, err := sb.Exec(ctx, []string{"mkdir", "-p", "/tmp/testdir"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = mkDir.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	snapshotImage, err := sb.SnapshotFilesystem(ctx, 55*time.Second)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(snapshotImage).ShouldNot(gomega.BeNil())
	g.Expect(snapshotImage.ImageID).To(gomega.HavePrefix("im-"))

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb2, err := tc.Sandboxes.Create(ctx, app, snapshotImage, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb2)

	proc, err := sb2.Exec(ctx, []string{"cat", "/tmp/test.txt"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(proc.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("test content"))

	dirCheck, err := sb2.Exec(ctx, []string{"test", "-d", "/tmp/testdir"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := dirCheck.Wait(ctx)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}
