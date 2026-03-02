package test

import (
	"bytes"
	"context"
	"io"
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func createSandbox(ctx context.Context, g *gomega.WithT, tc *modal.Client) *modal.Sandbox {
	g.THelper()

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb.SandboxID).ShouldNot(gomega.BeEmpty())
	return sb
}

func TestSandboxWriteAndReadBinaryFile(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)
	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	writer, err := sb.Open(ctx, "/tmp/test.bin", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	text := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	n, err := writer.Write(text)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(text)))
	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader, err := sb.Open(ctx, "/tmp/test.bin", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output := make([]byte, 10)
	n, err = reader.Read(output)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(10))
	g.Expect(output).Should(gomega.Equal(text))

	err = reader.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxAppendToFileBinary(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	g := gomega.NewWithT(t)
	c := newTestClient(t)
	sb := createSandbox(ctx, g, c)
	defer terminateSandbox(g, sb)

	writer, err := sb.Open(ctx, "/tmp/append.txt", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	text := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	n, err := writer.Write(text)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(text)))
	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	appender, err := sb.Open(ctx, "/tmp/append.txt", "a")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	moreText := []byte{7, 8, 9, 10}
	n, err = appender.Write(moreText)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(moreText)))

	reader, err := sb.Open(ctx, "/tmp/append.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	expectedText := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 8, 9, 10}
	out, err := io.ReadAll(reader)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(out).Should(gomega.Equal(expectedText))

	err = reader.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxFileFlush(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	g := gomega.NewWithT(t)
	c := newTestClient(t)
	sb := createSandbox(ctx, g, c)
	defer terminateSandbox(g, sb)

	writer, err := sb.Open(ctx, "/tmp/flush.txt", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	text := []byte("Test data")
	n, err := writer.Write(text)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(text)))
	err = writer.Flush()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader, err := sb.Open(ctx, "/tmp/flush.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	out, err := io.ReadAll(reader)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(out).Should(gomega.Equal(text))

	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = reader.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxMultipleFileOperations(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	g := gomega.NewWithT(t)
	c := newTestClient(t)
	sb := createSandbox(ctx, g, c)
	defer terminateSandbox(g, sb)

	content1 := []byte("File 1 content")
	writer, err := sb.Open(ctx, "/tmp/file1.txt", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	n, err := writer.Write(content1)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(content1)))
	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	content2 := []byte("File 2 content")
	writer, err = sb.Open(ctx, "/tmp/file2.txt", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	n, err = writer.Write(content2)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(content2)))
	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader1, err := sb.Open(ctx, "/tmp/file1.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	readContent1, err := io.ReadAll(reader1)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	err = reader1.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader2, err := sb.Open(ctx, "/tmp/file2.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	readContent2, err := io.ReadAll(reader2)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	err = reader2.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(readContent1).Should(gomega.Equal(content1))
	g.Expect(readContent2).Should(gomega.Equal(content2))
}

func TestSandboxFileOpenModes(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	g := gomega.NewWithT(t)
	c := newTestClient(t)
	sb := createSandbox(ctx, g, c)
	defer terminateSandbox(g, sb)

	content1 := []byte("Initial content")
	writer, err := sb.Open(ctx, "/tmp/modes.txt", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	n, err := writer.Write(content1)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(content1)))
	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader1, err := sb.Open(ctx, "/tmp/modes.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	readContent1, err := io.ReadAll(reader1)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(readContent1).Should(gomega.Equal(content1))
	err = reader1.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	content2 := []byte(" appended")
	appender, err := sb.Open(ctx, "/tmp/modes.txt", "a")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	n, err = appender.Write(content2)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(content2)))
	err = appender.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader2, err := sb.Open(ctx, "/tmp/modes.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	readContent2, err := io.ReadAll(reader2)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	expectedContent := []byte("Initial content appended")
	g.Expect(readContent2).Should(gomega.Equal(expectedContent))
	err = reader2.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxLargeFileOperations(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	g := gomega.NewWithT(t)
	c := newTestClient(t)
	sb := createSandbox(ctx, g, c)
	defer terminateSandbox(g, sb)

	xByte := []byte{'x'}
	largeData := bytes.Repeat(xByte, 1000)

	writer, err := sb.Open(ctx, "/tmp/large.txt", "w")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	n, err := writer.Write(largeData)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(n).Should(gomega.Equal(len(largeData)))
	err = writer.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	reader1, err := sb.Open(ctx, "/tmp/large.txt", "r")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	readContent1, err := io.ReadAll(reader1)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(readContent1).Should(gomega.Equal(largeData))
	err = reader1.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}
