package test

import (
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"go.uber.org/goleak"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// A V1 Sandbox's output comes from the control plane, not from the worker, so
// none of the Sandbox connection's own release machinery covers it: the stream
// sits on the long-lived control-plane connection, which nothing closes.
//
// The control plane here is a real server on a real connection on purpose.
// grpc-go runs a goroutine per client stream, and that goroutine is what a
// forgotten stream leaves behind; a hand-written stub standing in for the
// stream has no such goroutine, and a test written against one passes whether
// or not the leak is there.

// V1-shaped, so the Sandbox reads its output through the control plane.
const logsSandboxID = "sb-nGEijt9WbBMlGrsPH9FOaC"

type fakeLogsControlPlane struct {
	pb.UnimplementedModalClientServer
	line        string
	nextLine    string
	lastEntryID atomic.Value
	// When set, every call is failed with CANCELED instead of served, standing
	// in for a server that keeps cancelling.
	cancelEveryCall bool
	calls           atomic.Int64
	// Calls being served right now, which is what a leaked stream keeps alive.
	live atomic.Int64
}

// AppGetOrCreate gives the test a cheap call to open the connection with,
// before it starts counting goroutines.
func (s *fakeLogsControlPlane) AppGetOrCreate(
	_ context.Context, req *pb.AppGetOrCreateRequest,
) (*pb.AppGetOrCreateResponse, error) {
	return pb.AppGetOrCreateResponse_builder{AppId: req.GetAppName()}.Build(), nil
}

// SandboxGetLogs sends one batch and then holds the stream open, as the real
// control plane does while it waits for a Sandbox to print something more.
func (s *fakeLogsControlPlane) SandboxGetLogs(
	req *pb.SandboxGetLogsRequest, stream grpc.ServerStreamingServer[pb.TaskLogsBatch],
) error {
	s.calls.Add(1)
	s.lastEntryID.Store(req.GetLastEntryId())
	if s.cancelEveryCall {
		return status.Error(codes.Canceled, "cancelled by the server")
	}
	s.live.Add(1)
	defer s.live.Add(-1)
	entryID, line := "1-0", s.line
	switch req.GetLastEntryId() {
	case "0-0":
	case "1-0":
		entryID, line = "2-0", s.nextLine
	default:
		return status.Error(codes.InvalidArgument, "unexpected log cursor")
	}
	batch := pb.TaskLogsBatch_builder{
		EntryId: entryID,
		Items:   []*pb.TaskLogs{pb.TaskLogs_builder{Data: line}.Build()},
	}.Build()
	if err := stream.Send(batch); err != nil {
		return err
	}
	<-stream.Context().Done()
	return stream.Context().Err()
}

func startFakeLogsControlPlane(t *testing.T, line string) *modal.Client {
	t.Helper()
	client, _ := startFakeLogsControlPlaneWith(t, &fakeLogsControlPlane{line: line})
	return client
}

func startFakeLogsControlPlaneWith(t *testing.T, plane *fakeLogsControlPlane) (*modal.Client, *fakeLogsControlPlane) {
	t.Helper()
	g := gomega.NewWithT(t)

	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	pb.RegisterModalClientServer(server, plane)
	go func() {
		if err := server.Serve(listener); err != nil {
			t.Logf("fake control plane stopped: %v", err)
		}
	}()

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	client, err := modal.NewClientWithOptions(&modal.ClientParams{
		TokenID:            "test-token-id",
		TokenSecret:        "test-token-secret",
		Environment:        "test",
		ControlPlaneClient: pb.NewModalClientClient(conn),
		ControlPlaneConn:   conn,
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	t.Cleanup(func() {
		client.Close()
		_ = conn.Close()
		server.Stop()
		_ = listener.Close()
	})
	return client, plane
}

// Closing a partly read V1 output stream must release its gRPC goroutine.
func TestSandboxLogsCloseLeavesNoGoroutinesBehind(t *testing.T) {
	g := gomega.NewWithT(t)

	const line = "hello from the sandbox\n"
	client := startFakeLogsControlPlane(t, line)

	// Open the connection first: its transport goroutines belong to the client,
	// not to the stream this test is watching.
	_, err := client.Apps.FromName(t.Context(), "warmup", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	existing := goleak.IgnoreCurrent()

	sandbox, err := client.Sandboxes.FromID(t.Context(), logsSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	buf := make([]byte, len(line))
	n, err := sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf[:n])).To(gomega.Equal(line))

	g.Expect(sandbox.Stdout.Close()).To(gomega.Succeed())
	g.Eventually(func() error { return goleak.Find(existing) }, 8*time.Second, 50*time.Millisecond).
		Should(gomega.Succeed(), "closing a Sandbox output stream should release its goroutines")
}

// A server that keeps cancelling must exhaust the read's retry budget.
func TestServerCancellationDoesNotReopenForever(t *testing.T) {
	g := gomega.NewWithT(t)

	client, plane := startFakeLogsControlPlaneWith(t, &fakeLogsControlPlane{cancelEveryCall: true})
	sandbox, err := client.Sandboxes.FromID(t.Context(), logsSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	read := make(chan error, 1)
	go func() {
		_, readErr := sandbox.Stdout.Read(make([]byte, 64))
		read <- readErr
	}()

	g.Eventually(read, 30*time.Second).Should(gomega.Receive(gomega.HaveOccurred()),
		"a read against a server that keeps cancelling should end in an error")
	// The retry budget is ten, so the call count must remain bounded.
	g.Expect(plane.calls.Load()).To(gomega.BeNumerically("<=", int64(20)))
}

// V1 output uses the shared control-plane client and remains usable after detach.
func TestDetachLeavesV1OutputReadable(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	client, plane := startFakeLogsControlPlaneWith(t, &fakeLogsControlPlane{line: "hello\n"})
	sandbox, err := client.Sandboxes.FromID(t.Context(), logsSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = sandbox.Stdout.Close() })

	buf := make([]byte, 3)
	_, err = sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf)).To(gomega.Equal("hel"))
	g.Expect(sandbox.Detach()).To(gomega.Succeed())
	_, err = sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf)).To(gomega.Equal("lo\n"))
	g.Expect(plane.live.Load()).To(gomega.Equal(int64(1)))
	g.Expect(sandbox.Stdout.Close()).To(gomega.Succeed())
	g.Eventually(plane.live.Load).Should(gomega.BeZero())

	unread, err := client.Sandboxes.FromID(t.Context(), logsSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = unread.Stdout.Close() })
	g.Expect(unread.Detach()).To(gomega.Succeed())
	g.Expect(plane.calls.Load()).To(gomega.Equal(int64(1)))
	_, err = unread.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf)).To(gomega.Equal("hel"))
	g.Expect(plane.calls.Load()).To(gomega.Equal(int64(2)))
}

// Reading part of a V1 Sandbox's output and then forgetting it must give the
// stream back, without a Close and without a Detach.
func TestSandboxLogsPartialReadLeavesNoGoroutinesBehind(t *testing.T) {
	g := gomega.NewWithT(t)

	const line = "hello from the sandbox\n"
	client := startFakeLogsControlPlane(t, line)

	// Open the connection first: its transport goroutines belong to the client,
	// not to the stream this test is watching.
	_, err := client.Apps.FromName(t.Context(), "warmup", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	existing := goleak.IgnoreCurrent()

	sandbox, err := client.Sandboxes.FromID(t.Context(), logsSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	buf := make([]byte, len(line))
	n, err := sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf[:n])).To(gomega.Equal(line))

	// Walk away: no Close, no Detach. The stream is still open at this point,
	// and the idle release is what has to end it.
	g.Eventually(func() error { return goleak.Find(existing) }, 8*time.Second, 50*time.Millisecond).
		Should(gomega.Succeed(), "a Sandbox output stream nobody is reading should leave no goroutines behind")
}

// And the reader must still work afterwards: the next read reopens the stream
// and carries on from where the last one stopped.
func TestSandboxLogsReadAgainAfterIdleRelease(t *testing.T) {
	g := gomega.NewWithT(t)

	const line = "hello from the sandbox\n"
	const nextLine = "subsequent output\n"
	client, plane := startFakeLogsControlPlaneWith(t, &fakeLogsControlPlane{line: line, nextLine: nextLine})

	sandbox, err := client.Sandboxes.FromID(t.Context(), logsSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	t.Cleanup(func() { _ = sandbox.Stdout.Close() })

	buf := make([]byte, len(line))
	n, err := sandbox.Stdout.Read(buf[:5])
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf[:n])).To(gomega.Equal(line[:5]))
	g.Expect(plane.lastEntryID.Load()).To(gomega.Equal("0-0"))

	// Wait for idle cancellation before consuming the rest of the buffered batch.
	g.Eventually(plane.live.Load, 8*time.Second).Should(gomega.BeZero())
	n, err = sandbox.Stdout.Read(buf)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(string(buf[:n])).To(gomega.Equal(line[5:]))
	g.Expect(plane.calls.Load()).To(gomega.Equal(int64(1)))

	// Read from a goroutine: a stream that was not given back leaves this read
	// waiting on a batch that never comes, and the test should say so rather
	// than hang until the package timeout.
	type readResult struct {
		text string
		err  error
	}
	done := make(chan readResult, 1)
	go func() {
		n, readErr := sandbox.Stdout.Read(buf)
		done <- readResult{text: string(buf[:n]), err: readErr}
	}()

	var got readResult
	g.Eventually(done, 10*time.Second).Should(gomega.Receive(&got),
		"a released stream should reopen on the next read")
	g.Expect(got.err).ToNot(gomega.HaveOccurred())
	g.Expect(got.text).To(gomega.Equal(nextLine))
	g.Expect(plane.lastEntryID.Load()).To(gomega.Equal("1-0"))
	g.Expect(plane.calls.Load()).To(gomega.Equal(int64(2)))

	g.Expect(sandbox.Stdout.Close()).To(gomega.Succeed())
}
