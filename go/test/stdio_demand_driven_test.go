package test

import (
	"io"
	"runtime"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

// Exec output is pulled by the read that wants it. Nothing runs on the caller's
// behalf between reads, so a caller who reads part of the output and stops
// leaves nothing behind - no goroutine, and nothing read ahead beyond the chunk
// they were handed.

// settled gives whatever the last call started a moment to be scheduled, so a
// count taken after it is not merely a count taken too early.
func settled() int {
	for range 5 {
		runtime.Gosched()
		time.Sleep(20 * time.Millisecond)
	}
	return runtime.NumGoroutine()
}

// The read is what opens the stream: until one happens the worker has been
// asked for nothing.
func TestExecOutputIsRequestedByTheFirstRead(t *testing.T) {
	g := gomega.NewWithT(t)

	router, _, sandbox := startFakeWorker(t)

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	before := settled()
	// Reaching for the field is not reading it.
	stdout := process.Stdout
	g.Expect(router.requestedOffsets()).To(gomega.BeEmpty(),
		"no read has happened, so the worker should not have been asked for output")
	g.Expect(settled()).To(gomega.Equal(before),
		"reaching for Stdout should not start anything")

	buf := make([]byte, 7)
	n, err := stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.BeNumerically(">", 0))
	g.Expect(router.requestedOffsets()).To(gomega.Equal([]uint64{0}),
		"the first read should have opened the stream")
}

// Same for a Sandbox's own logs, which come from the control plane rather than
// the worker.
func TestSandboxLogsAreRequestedByTheFirstRead(t *testing.T) {
	g := gomega.NewWithT(t)

	_, _, sandbox := startFakeWorker(t)

	before := settled()
	stdout := sandbox.Stdout
	g.Expect(stdout).ToNot(gomega.BeNil())
	g.Expect(settled()).To(gomega.Equal(before),
		"reaching for Sandbox.Stdout should not start anything")
}

// A caller who reads it all still gets it all, over one stream.
func TestExecOutputArrivesInFull(t *testing.T) {
	g := gomega.NewWithT(t)

	router, _, sandbox := startFakeWorker(t)

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer func() { _ = process.Stdout.Close() }()

	// Exactly the output: the worker holds the stream open afterwards, as it
	// does for an exec still running, so reading to EOF would never return.
	got := make([]byte, len(fakeWorkerOutput))
	_, err = io.ReadFull(process.Stdout, got)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(got).To(gomega.Equal(fakeWorkerOutput))
	g.Expect(router.requestedOffsets()).To(gomega.Equal([]uint64{0}),
		"a prompt reader gives no reason to reopen the stream")
}

// Closing a partly read stream puts back everything reading it started.
func TestClosingAPartlyReadStreamLeavesNothingRunning(t *testing.T) {
	g := gomega.NewWithT(t)

	_, listener, sandbox := startFakeWorker(t)

	process, err := sandbox.Exec(t.Context(), []string{"echo", "hi"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	before := settled()
	buf := make([]byte, 7)
	_, err = process.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(process.Stdout.Close()).To(gomega.Succeed())
	g.Expect(settled()).To(gomega.Equal(before),
		"closing should put back whatever reading started")
	// The connection is a separate matter: nothing here gives it up.
	g.Expect(listener.Live()).To(gomega.Equal(1))
}

// Close has to reach a Read that is already blocked on the stream, since that
// is how a caller stops one: SandboxFilesystem.Watch closes stdout from a
// context.AfterFunc precisely to unblock a read.
func TestClosingExecStdioUnblocksAWaitingRead(t *testing.T) {
	g := gomega.NewWithT(t)

	// An exec that has printed nothing: the stream opens and stays open, so the
	// read below has nothing to return until something ends it.
	router, _, sandbox := startFakeWorkerPrinting(t, nil)

	process, err := sandbox.Exec(t.Context(), []string{"sleep", "60"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	readErr := make(chan error, 1)
	go func() {
		_, err := process.Stdout.Read(make([]byte, 1))
		readErr <- err
	}()

	// Close must land while the read is blocked. The worker seeing the stream
	// open says the read got that far; the pause gives it time to reach Recv.
	g.Eventually(router.requestedOffsets, 5*time.Second).ShouldNot(gomega.BeEmpty())
	time.Sleep(100 * time.Millisecond)

	closeErr := make(chan error, 1)
	go func() { closeErr <- process.Stdout.Close() }()

	g.Eventually(closeErr, 5*time.Second).Should(gomega.Receive(gomega.BeNil()),
		"Close must not wait for the read it is cancelling")
	g.Eventually(readErr, 5*time.Second).Should(gomega.Receive(gomega.HaveOccurred()),
		"the blocked read should end once the stream is cancelled")
}
