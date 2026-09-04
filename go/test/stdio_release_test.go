package test

import (
	"io"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"go.uber.org/goleak"
)

// Giving the connection back when nobody is reading, entered where callers
// enter. These count connections at the worker rather than reading the client's
// own state, so they say what an operator would see.

// Short enough to keep tests quick. One timeout governs the whole client, so a
// release lands a little after it.
const releaseIdleTimeout = time.Second

func releasingWorkerOpts(output []byte) fakeWorkerOpts {
	return fakeWorkerOpts{
		output:             output,
		channelIdleTimeout: "1",
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
	g.Expect(time.Since(start)).To(gomega.BeNumerically(">=", releaseIdleTimeout/2),
		"released well before the idle timeout was up")
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

// Zero turns the release off, as the config documents: the connection stays up
// however long the caller takes.
func TestSandboxExecKeepsConnectionWhenIdleTimeoutIsZero(t *testing.T) {
	g := gomega.NewWithT(t)

	router, listener, sandbox := startFakeWorkerWith(t, fakeWorkerOpts{
		channelIdleTimeout: "0",
	})
	// Opting out of the release means owning the cleanup: nothing else will
	// give this connection back.
	t.Cleanup(func() { _ = sandbox.Detach() })

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	buf := make([]byte, 7)
	_, err = process.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Consistently(listener.Live, 3*releaseIdleTimeout, 50*time.Millisecond).Should(
		gomega.Equal(1), "zero should keep the connection up",
	)
	g.Expect(router.requestedOffsets()).To(gomega.Equal([]uint64{0}),
		"the stream should not have reopened")
}

// A reader that keeps coming back keeps the connection: each read refreshes the
// timer, so gaps shorter than it never add up to a release however long the
// whole read takes.
func TestSandboxExecSteadyReaderRefreshesTheIdleTimer(t *testing.T) {
	g := gomega.NewWithT(t)

	router, listener, sandbox := startFakeWorkerWith(t, fakeWorkerOpts{
		channelIdleTimeout: "1",
	})

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	// Five 7-byte chunks read 400ms apart: every gap is inside the timeout, but
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
		"every read refreshes the timer, so the stream should never have reopened")
	g.Expect(listener.Live()).To(gomega.Equal(1))
}

// The point of releasing on idle is that forgetting to Detach costs nothing.
// Not just the socket: the connection object and the goroutines behind it go
// too, which is what a leak check would otherwise catch much later.
func TestIdleSandboxLeavesNoGoroutinesBehind(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))
	// Only goroutines this test starts count: others are still giving their
	// connections back while it runs.
	existing := goleak.IgnoreCurrent()

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	_, err = process.Stdout.Read(make([]byte, 7))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(listener.Live()).To(gomega.Equal(1))

	// Walk away: no Detach, no Close.
	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "the socket should have gone")
	g.Eventually(func() error { return goleak.Find(existing) }, 5*time.Second, 50*time.Millisecond).
		Should(gomega.Succeed(), "an idle Sandbox should leave no goroutines behind either")
}

// And it must still be usable afterwards: the next read dials again and picks
// up where it left off, without spending the reader's retry budget.
func TestIdleSandboxReconnectsWhenReadAgain(t *testing.T) {
	g := gomega.NewWithT(t)

	opts := releasingWorkerOpts(nil)
	// A chunk per stream, so the read after the release has to reopen.
	opts.chunksPerStream = 1
	router, listener, sandbox := startFakeWorkerWith(t, opts)

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	got := make([]byte, 0, len(fakeWorkerOutput))
	for round := range 3 {
		chunk := make([]byte, 7)
		_, err := io.ReadFull(process.Stdout, chunk)
		g.Expect(err).ToNot(gomega.HaveOccurred(), "round %d", round)
		got = append(got, chunk...)

		g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
			gomega.Equal(0), "round %d: the connection should have been given up", round)
	}

	g.Expect(got).To(gomega.Equal(fakeWorkerOutput[:len(got)]),
		"output across reconnects must be continuous - no repeats and no gaps")
	g.Expect(router.requestedOffsets()).To(gomega.Equal([]uint64{0, 7, 14}))
}

// Detaching ends a Sandbox for good: unlike an idle release, nothing picks it
// up again.
func TestDetachStillEndsTheSandboxForGood(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	_, err = process.Stdout.Read(make([]byte, 7))
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(sandbox.Detach()).To(gomega.Succeed())
	g.Eventually(listener.Live, 4*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "detaching should give the connection back at once")

	// Unlike an idle release, nothing reconnects afterwards.
	_, err = process.Stdout.Read(make([]byte, 7))
	g.Expect(err).To(gomega.HaveOccurred(), "reading a detached Sandbox must fail")
	_, err = sandbox.Exec(t.Context(), []string{"echo", "again"}, nil)
	g.Expect(err).To(gomega.HaveOccurred(), "using a detached Sandbox must fail")
}

// Detaching after the connection has already gone back is not an error, and
// still ends the Sandbox. A release leaves the client able to dial again, so
// only being closed keeps the next operation from quietly reconnecting.
func TestDetachAfterAnIdleReleaseStillEndsTheSandbox(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	_, err = process.Stdout.Read(make([]byte, 7))
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "the connection should have been given up")

	g.Expect(sandbox.Detach()).To(gomega.Succeed(),
		"detaching a Sandbox whose connection already went back is not an error")

	_, err = sandbox.Exec(t.Context(), []string{"echo", "again"}, nil)
	g.Expect(err).To(gomega.HaveOccurred(), "using a detached Sandbox must fail")
	g.Expect(listener.Live()).To(gomega.Equal(0), "and must not have dialled again")
}

// A release already scheduled when the Sandbox is detached has to come to
// nothing. The callback runs on its own goroutine, so a panic there
// takes the process down rather than failing an assertion.
func TestIdleReleaseScheduledBeforeADetachIsHarmless(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	// Arms the timer, then detaches well inside it.
	_, err = process.Stdout.Read(make([]byte, 7))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(sandbox.Detach()).To(gomega.Succeed())

	// Sit through the release the read had scheduled.
	time.Sleep(3 * releaseIdleTimeout)

	g.Expect(listener.Live()).To(gomega.Equal(0))
	_, err = sandbox.Exec(t.Context(), []string{"echo", "again"}, nil)
	g.Expect(err).To(gomega.HaveOccurred(), "the Sandbox should still be detached")
}

// Not every operation goes through the same helper, so the lease is taken where
// they all meet. One that takes a different route must still find a connection.
func TestReleasedConnectionIsRebuiltByAnyOperation(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	_, err = process.Stdout.Read(make([]byte, 7))
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "the connection should have been given up")

	// ReloadVolumes reaches the worker by a different path than exec does. The
	// fake does not implement it, so an answer from the worker - any answer -
	// is what says the connection was rebuilt rather than left absent.
	err = sandbox.ReloadVolumes(t.Context(), nil)
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("Unimplemented")),
		"the call should have reached the worker")
	g.Expect(listener.Live()).To(gomega.Equal(1))
}

// A V2 Sandbox's own output comes from the command router, so it holds a
// connection the same way an exec's does.
func TestSandboxStdoutReleasesConnection(t *testing.T) {
	g := gomega.NewWithT(t)

	router, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))

	buf := make([]byte, 7)
	n, err := sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.BeNumerically(">", 0))
	g.Expect(router.sandboxRequestedOffsets()).To(gomega.Equal([]uint64{0}),
		"the read should have opened a Sandbox stdio stream")
	g.Expect(listener.Live()).To(gomega.Equal(1))

	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "a Sandbox nobody is reading should not hold a connection")
}

// And reading it again picks up where it left off.
func TestSandboxStdoutResumesAfterConnectionReleased(t *testing.T) {
	g := gomega.NewWithT(t)

	opts := releasingWorkerOpts(nil)
	// A chunk per stream, so each read after a release has to reopen.
	opts.chunksPerStream = 1
	router, listener, sandbox := startFakeWorkerWith(t, opts)

	var got []byte
	for round := range 3 {
		chunk := make([]byte, 7)
		_, err := io.ReadFull(sandbox.Stdout, chunk)
		g.Expect(err).ToNot(gomega.HaveOccurred(), "round %d", round)
		got = append(got, chunk...)

		g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
			gomega.Equal(0), "round %d: the connection should have been given up", round)
	}

	g.Expect(got).To(gomega.Equal(fakeWorkerOutput[:len(got)]),
		"output across reconnects must be continuous - no repeats and no gaps")
	g.Expect(router.sandboxRequestedOffsets()).To(gomega.Equal([]uint64{0, 7, 14}))
}

// Terminating a Sandbox leaves the caller attached, so its output is still
// readable - and the connection still goes back once nothing is using it.
func TestTerminatedSandboxStaysReadableThenReleases(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorkerWith(t, releasingWorkerOpts(nil))
	// Only goroutines this test starts count.
	existing := goleak.IgnoreCurrent()

	_, err := sandbox.Terminate(t.Context(), nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	// Still attached, so the output a terminated Sandbox produced is readable.
	buf := make([]byte, 7)
	n, err := sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred(), "a terminated Sandbox should still be readable")
	g.Expect(n).To(gomega.BeNumerically(">", 0))

	// And nothing is held once the caller stops reading.
	g.Eventually(listener.Live, 8*releaseIdleTimeout, 20*time.Millisecond).Should(
		gomega.Equal(0), "a terminated Sandbox nobody is reading should not hold a connection")
	g.Eventually(func() error { return goleak.Find(existing) }, 5*time.Second, 50*time.Millisecond).
		Should(gomega.Succeed(), "and should leave no goroutines behind either")
}
