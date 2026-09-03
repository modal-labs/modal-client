package test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"sync"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// A local worker over a real socket, with the control plane mocked, entered
// where callers enter: Sandbox.Exec. Tests that care what happens between the
// caller and the wire need to start here, since that is exactly what a test
// starting further down cannot see.

const (
	fakeWorkerSandboxID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
	fakeWorkerTaskID    = "ta-01ARZ3NDEKTSV4RRFFQ69G5FAV"
)

var fakeWorkerOutput = []byte("line-0\nline-1\nline-2\nline-3\nline-4\n")

// fakeWorkerRouter serves exec stdio from a byte offset, the way a worker seeks
// its stdio file, and records the offset each read asked for.
type fakeWorkerRouter struct {
	pb.UnimplementedTaskCommandRouterServer
	// What the exec prints, in full. Once it has all been sent the stream stays
	// open with nothing more on it, as it does for an exec still running.
	output []byte
	// When set, how many chunks one stream will serve before it stops sending
	// and simply holds open. A reader wanting more has to reopen, which is what
	// makes resuming at an offset observable.
	chunksPerStream int
	mu              sync.Mutex
	offsets         []uint64
}

func (s *fakeWorkerRouter) TaskExecStart(
	ctx context.Context,
	req *pb.TaskExecStartRequest,
) (*pb.TaskExecStartResponse, error) {
	return pb.TaskExecStartResponse_builder{}.Build(), nil
}

func (s *fakeWorkerRouter) TaskExecStdioRead(
	req *pb.TaskExecStdioReadRequest,
	stream grpc.ServerStreamingServer[pb.TaskExecStdioReadResponse],
) error {
	offset := req.GetOffset()
	s.mu.Lock()
	s.offsets = append(s.offsets, offset)
	s.mu.Unlock()

	sent := 0
	for offset < uint64(len(s.output)) {
		if s.chunksPerStream > 0 && sent == s.chunksPerStream {
			break
		}
		sent++
		end := min(offset+7, uint64(len(s.output)))
		if err := stream.Send(pb.TaskExecStdioReadResponse_builder{
			Data: s.output[offset:end],
		}.Build()); err != nil {
			return err
		}
		offset = end
	}
	// Hold the stream open, as a worker does for an exec still running.
	<-stream.Context().Done()
	return stream.Context().Err()
}

func (s *fakeWorkerRouter) requestedOffsets() []uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]uint64(nil), s.offsets...)
}

// countingListener reports how many connections the server currently holds,
// which is the only way to tell a released connection from a quiet one.
type countingListener struct {
	net.Listener
	mu   sync.Mutex
	live int
}

func (l *countingListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	l.mu.Lock()
	l.live++
	l.mu.Unlock()
	return &countingConn{Conn: conn, listener: l}, nil
}

func (l *countingListener) Live() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.live
}

type countingConn struct {
	net.Conn
	listener *countingListener
	once     sync.Once
}

func (c *countingConn) Close() error {
	c.once.Do(func() {
		c.listener.mu.Lock()
		c.listener.live--
		c.listener.mu.Unlock()
	})
	return c.Conn.Close()
}

func mockRouterJWT() string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	payload, _ := json.Marshal(map[string]any{"exp": time.Now().Unix() + 3600})
	return header + "." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
}

// startFakeWorker serves a router on a real socket and points a mocked control
// plane at it, returning a Sandbox reached the way a caller reaches one.
func startFakeWorker(t *testing.T) (*fakeWorkerRouter, *countingListener, *modal.Sandbox) {
	t.Helper()
	return startFakeWorkerWith(t, fakeWorkerOpts{})
}

// startFakeWorkerPrinting is startFakeWorker for a test that needs to say what
// the exec prints - notably how many chunks it arrives in.
func startFakeWorkerPrinting(t *testing.T, output []byte) (*fakeWorkerRouter, *countingListener, *modal.Sandbox) {
	t.Helper()
	return startFakeWorkerWith(t, fakeWorkerOpts{output: output})
}

// fakeWorkerOpts is what a test wants to vary about the worker it talks to.
// A zero field takes the default: fakeWorkerOutput, and whatever timeouts the
// SDK ships with.
type fakeWorkerOpts struct {
	output []byte
	// Written to the environment as the SDK reads it, so "0" is meaningfully
	// different from unset.
	channelIdleTimeout string
	// Chunks one stream will serve before it holds open without sending more.
	chunksPerStream int
}

func startFakeWorkerWith(t *testing.T, opts fakeWorkerOpts) (*fakeWorkerRouter, *countingListener, *modal.Sandbox) {
	t.Helper()
	g := gomega.NewWithT(t)

	output := opts.output
	if output == nil {
		output = fakeWorkerOutput
	}

	if opts.channelIdleTimeout != "" {
		t.Setenv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", opts.channelIdleTimeout)
	}

	// Read when the client builds its profile, so both must be set first. A
	// localhost server URL is what makes the SDK dial the router without TLS.
	t.Setenv("MODAL_SERVER_URL", "http://127.0.0.1:1")

	router := &fakeWorkerRouter{output: output, chunksPerStream: opts.chunksPerStream}
	raw, err := net.Listen("tcp", "127.0.0.1:0")
	g.Expect(err).ToNot(gomega.HaveOccurred())
	listener := &countingListener{Listener: raw}

	server := grpc.NewServer()
	pb.RegisterTaskCommandRouterServer(server, router)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)

	routerURL := "https://" + raw.Addr().String()
	mock := newGRPCMockClient(t)
	grpcmock.HandleUnary(mock, "SandboxGetTaskIdV2",
		func(req *pb.SandboxGetTaskIdRequest) (*pb.SandboxGetTaskIdResponse, error) {
			return pb.SandboxGetTaskIdResponse_builder{TaskId: proto.String(fakeWorkerTaskID)}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxGetCommandRouterAccess",
		func(req *pb.SandboxGetCommandRouterAccessRequest) (*pb.SandboxGetCommandRouterAccessResponse, error) {
			return pb.SandboxGetCommandRouterAccessResponse_builder{
				Url: routerURL,
				Jwt: mockRouterJWT(),
			}.Build(), nil
		})

	sandbox, err := mock.Sandboxes.FromID(t.Context(), fakeWorkerSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	// Detaching closes the worker connection the Sandbox opened, which the
	// package's goroutine-leak check would otherwise flag.
	t.Cleanup(func() { _ = sandbox.Detach() })
	return router, listener, sandbox
}
