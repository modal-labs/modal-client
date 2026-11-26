package test

import (
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
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

func TestSandboxLogsReuseImplMetadataMock(t *testing.T) {
	for _, v2 := range []bool{false, true} {
		for _, path := range []string{"create", "create-fallback", "name", "list", "restore", "lazy", "lazy-error"} {
			suffix := ""
			sandboxID := logsSandboxID
			if v2 {
				suffix = "V2"
				sandboxID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
			}
			t.Run(suffix+"/"+path, func(t *testing.T) {
				backend := "0"
				if v2 {
					backend = "1"
				}
				t.Setenv("MODAL_SANDBOX_V2", backend)
				g := gomega.NewWithT(t)
				ctx := t.Context()
				mock := newGRPCMockClient(t)
				metadata := pb.SandboxHandleMetadata_builder{
					AppId:  "ap-known",
					Result: pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_TERMINATED}.Build(),
				}.Build()
				fallback := path == "lazy" || path == "lazy-error" || (path == "restore" && !v2)
				taskID := "ta-known"
				handleTaskID := func() {
					grpcmock.HandleUnary(mock, "SandboxGetTaskId"+suffix, func(*pb.SandboxGetTaskIdRequest) (*pb.SandboxGetTaskIdResponse, error) {
						return pb.SandboxGetTaskIdResponse_builder{TaskId: &taskID}.Build(), nil
					})
				}
				var sb *modal.Sandbox
				var err error
				switch path {
				case "create", "create-fallback":
					createMetadata := metadata
					if path == "create-fallback" {
						createMetadata = nil
					}
					if v2 {
						grpcmock.HandleUnary(mock, "SandboxCreateV2", func(*pb.SandboxCreateV2Request) (*pb.SandboxCreateV2Response, error) {
							return pb.SandboxCreateV2Response_builder{SandboxId: sandboxID, Metadata: createMetadata}.Build(), nil
						})
						sb, err = mock.Sandboxes.Create(ctx, &modal.App{AppID: "ap-create"}, &modal.Image{ImageID: "im-known"}, nil)
					} else {
						grpcmock.HandleUnary(mock, "SandboxCreate", func(*pb.SandboxCreateRequest) (*pb.SandboxCreateResponse, error) {
							return pb.SandboxCreateResponse_builder{SandboxId: sandboxID, Metadata: createMetadata}.Build(), nil
						})
						handleTaskID()
						sb, err = mock.Sandboxes.Create(ctx, &modal.App{AppID: "ap-create"}, &modal.Image{ImageID: "im-known"}, nil)
					}
				case "list":
					grpcmock.HandleUnary(mock, "SandboxList"+suffix, func(*pb.SandboxListRequest) (*pb.SandboxListResponse, error) {
						return pb.SandboxListResponse_builder{Sandboxes: []*pb.SandboxInfo{
							pb.SandboxInfo_builder{Id: sandboxID, Metadata: metadata}.Build(),
						}}.Build(), nil
					})
					sandboxes, listErr := mock.Sandboxes.List(ctx, &modal.SandboxListParams{AppID: "ap-known"})
					g.Expect(listErr).NotTo(gomega.HaveOccurred())
					for sandbox, listErr := range sandboxes {
						sb, err = sandbox, listErr
						break
					}
				case "restore":
					grpcmock.HandleUnary(mock, "SandboxSnapshotGet", func(*pb.SandboxSnapshotGetRequest) (*pb.SandboxSnapshotGetResponse, error) {
						return pb.SandboxSnapshotGetResponse_builder{
							HandleMetadata: pb.SandboxSnapshotHandleMetadata_builder{IsV2: v2}.Build(),
						}.Build(), nil
					})
					if v2 {
						grpcmock.HandleUnary(mock, "SandboxRestoreV2", func(*pb.SandboxRestoreV2Request) (*pb.SandboxRestoreV2Response, error) {
							return pb.SandboxRestoreV2Response_builder{SandboxId: sandboxID, Metadata: metadata}.Build(), nil
						})
					} else {
						grpcmock.HandleUnary(mock, "SandboxRestore", func(*pb.SandboxRestoreRequest) (*pb.SandboxRestoreResponse, error) {
							return pb.SandboxRestoreResponse_builder{SandboxId: sandboxID}.Build(), nil
						})
						handleTaskID()
					}
					snapshot, snapshotErr := mock.SandboxSnapshots.FromID(ctx, "sn-01BX5ZZKBKACTAV9WEVGEMMVRY", nil)
					g.Expect(snapshotErr).NotTo(gomega.HaveOccurred())
					sb, err = mock.Sandboxes.ExperimentalFromSnapshot(ctx, snapshot, nil)
				default:
					nameMetadata := metadata
					if fallback {
						nameMetadata = nil
					}
					grpcmock.HandleUnary(mock, "SandboxGetFromName"+suffix, func(*pb.SandboxGetFromNameRequest) (*pb.SandboxGetFromNameResponse, error) {
						return pb.SandboxGetFromNameResponse_builder{SandboxId: sandboxID, Metadata: nameMetadata}.Build(), nil
					})
					sb, err = mock.Sandboxes.FromName(ctx, "libmodal-test", "sandbox", nil)
				}
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(sb).NotTo(gomega.BeNil())
				if fallback {
					grpcmock.HandleUnary(mock, "SandboxWait"+suffix, func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
						g.Expect(req.GetTimeout()).To(gomega.BeZero())
						if path == "lazy-error" {
							return nil, status.Error(codes.NotFound, "missing")
						}
						return pb.SandboxWaitResponse_builder{Metadata: metadata}.Build(), nil
					})
				} else if path != "create-fallback" {
					result, pollErr := sb.Poll(ctx, nil)
					g.Expect(pollErr).NotTo(gomega.HaveOccurred())
					g.Expect(result).NotTo(gomega.BeNil())
					g.Expect(*result).To(gomega.Equal(137))
					code, waitErr := sb.Wait(ctx, nil)
					g.Expect(waitErr).NotTo(gomega.HaveOccurred())
					g.Expect(code).To(gomega.Equal(137))
				}
				if path == "lazy-error" {
					_, err := sb.Logs.Tail(ctx, &modal.LogTailParams{Entries: 1})
					g.Expect(status.Code(err)).To(gomega.Equal(codes.NotFound))
				} else {
					handleTaskID()
					grpcmock.HandleUnary(mock, "AppFetchLogs", func(req *pb.AppFetchLogsRequest) (*pb.AppFetchLogsResponse, error) {
						appID := "ap-known"
						if path == "create-fallback" {
							appID = "ap-create"
						}
						g.Expect(req.GetAppId()).To(gomega.Equal(appID))
						return pb.AppFetchLogsResponse_builder{Batches: []*pb.TaskLogsBatch{
							pb.TaskLogsBatch_builder{TaskId: taskID, Items: []*pb.TaskLogs{
								pb.TaskLogs_builder{Data: "known log\n", Timestamp: float64(time.Now().Unix())}.Build(),
							}}.Build(),
						}}.Build(), nil
					})
					entries, err := sb.Logs.Tail(ctx, &modal.LogTailParams{Entries: 1})
					g.Expect(err).NotTo(gomega.HaveOccurred())
					var messages []string
					for entry, logErr := range entries {
						g.Expect(logErr).NotTo(gomega.HaveOccurred())
						messages = append(messages, entry.Message)
					}
					g.Expect(messages).To(gomega.Equal([]string{"known log\n"}))
					if fallback {
						result, pollErr := sb.Poll(ctx, nil)
						g.Expect(pollErr).NotTo(gomega.HaveOccurred())
						g.Expect(result).NotTo(gomega.BeNil())
						g.Expect(*result).To(gomega.Equal(137))
					}
				}
				g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
			})
		}
	}
}

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

func (s *fakeLogsControlPlane) SandboxWait(context.Context, *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
	return pb.SandboxWaitResponse_builder{}.Build(), nil
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
