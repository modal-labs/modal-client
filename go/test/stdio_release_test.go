package test

import (
	"io"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

// Giving the connection back when nobody is reading, entered where callers
// enter. These count connections at the worker rather than reading the client's
// own state, so they say what an operator would see.

const (
	// Short enough to keep tests quick. A release waits out the reader's grace
	// and then the channel timeout, so it lands a little after the two together.
	releaseIdleTimeout = time.Second
	releaseStreamIdle  = 100 * time.Millisecond
)

func releasingWorkerOpts(output []byte) fakeWorkerOpts {
	return fakeWorkerOpts{
		output:             output,
		channelIdleTimeout: "1",
		streamIdleTimeout:  "0.1",
	}
}

// Reading part of an exec's output and then forgetting the Sandbox must give
// the connection back.
func TestSandboxExecPartialReadReleasesConnection(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	buf := make([]byte, 7)
	n, err := process.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.BeNumerically(">", 0))
	g.Expect(listener.Live()).To(gomega.Equal(1))

	start := time.Now()
	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "an exec nobody is reading should not hold a connection",
	)
	g.Expect(time.Since(start)).To(gomega.BeNumerically(">=", releaseStreamIdle),
		"released before the reader's grace was up")
}

// An exec that prints once and then goes quiet while still running - a server
// logging that it is listening, say. There is no next chunk to wait on, so only
// the reader's silence can say that nobody is reading.
func TestSandboxExecReleasesWhenTheExecGoesQuiet(t *testing.T) {
	g := gomega.NewWithT(t)

	// Short enough to arrive as a single chunk.
	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts([]byte("up\n")))

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	buf := make([]byte, 8)
	n, err := process.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(buf[:n]).To(gomega.Equal([]byte("up\n")))

	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "a quiet exec nobody is reading should not hold a connection",
	)
}

// Coming back to a half-read exec must continue where it left off, however many
// times the connection is given up in between.
func TestSandboxExecResumesAfterConnectionReleased(t *testing.T) {
	g := gomega.NewWithT(t)

	opts := releasingWorkerOpts(nil)
	// A chunk per stream, so each read after a release has to reopen. Otherwise
	// the whole output arrives in the transport buffer at once and a reader
	// never needs to ask again.
	opts.chunksPerStream = 1
	router, listener, sandbox := startFakeWorkerWith(t, opts)

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	// Read a chunk, let the connection go, read the next. Three rounds is
	// enough to show it is not a one-off.
	var got []byte
	for round := range 3 {
		chunk := make([]byte, 7)
		_, err := io.ReadFull(process.Stdout, chunk)
		g.Expect(err).ToNot(gomega.HaveOccurred(), "round %d", round)
		got = append(got, chunk...)

		g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
			gomega.Equal(0), "round %d: the connection should have been given up", round,
		)
	}

	g.Expect(got).To(gomega.Equal(fakeWorkerOutput[:len(got)]),
		"resumed output must be continuous - no repeats and no gaps")

	// Every reopen asks for the byte after the ones already handed over.
	offsets := router.requestedOffsets()
	g.Expect(offsets).To(gomega.Equal([]uint64{0, 7, 14}))
}

// Zero turns the reader grace off, as the config documents: the stream stays
// open however long the caller takes.
func TestSandboxExecKeepsStreamWhenStreamIdleIsZero(t *testing.T) {
	g := gomega.NewWithT(t)

	router, listener, sandbox := startFakeWorkerWith(t, fakeWorkerOpts{
		channelIdleTimeout: "1",
		streamIdleTimeout:  "0",
	})

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	buf := make([]byte, 7)
	_, err = process.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Consistently(listener.Live, 3*releaseIdleTimeout, 50*time.Millisecond).Should(
		gomega.Equal(1), "zero should keep the connection while the stream is open",
	)
	g.Expect(router.requestedOffsets()).To(gomega.Equal([]uint64{0}),
		"the stream should not have reopened")
}

// A reader that keeps coming back keeps the stream: each read refreshes the
// grace, so gaps shorter than it never add up to a release however long the
// whole read takes.
func TestSandboxExecSteadyReaderRefreshesTheStreamGrace(t *testing.T) {
	g := gomega.NewWithT(t)

	router, listener, sandbox := startFakeWorkerWith(t, fakeWorkerOpts{
		channelIdleTimeout: "1",
		streamIdleTimeout:  "1",
	})

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	// Five 7-byte chunks read 400ms apart: every gap is inside the grace, but
	// the read as a whole runs to about twice it.
	got := make([]byte, 0, len(fakeWorkerOutput))
	buf := make([]byte, 7)
	for len(got) < len(fakeWorkerOutput) {
		n, err := io.ReadFull(process.Stdout, buf)
		g.Expect(err).ToNot(gomega.HaveOccurred())
		got = append(got, buf[:n]...)
		time.Sleep(400 * time.Millisecond)
	}

	g.Expect(got).To(gomega.Equal(fakeWorkerOutput))
	g.Expect(router.requestedOffsets()).To(gomega.Equal([]uint64{0}),
		"every read refreshes the grace, so the stream should never have reopened")
	g.Expect(listener.Live()).To(gomega.Equal(1))
}
