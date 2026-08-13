package modal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type mockLogsClient struct {
	pb.ModalClientClient
	mu sync.Mutex

	requests       []*pb.AppFetchLogsRequest
	countRequests  []*pb.AppCountLogsRequest
	infoRequests   []*pb.FunctionCallGetInfoRequest
	streamRequests []*pb.AppGetLogsRequest

	responder       func(context.Context, *pb.AppFetchLogsRequest, int) (*pb.AppFetchLogsResponse, error)
	countResponder  func(*pb.AppCountLogsRequest, int) (*pb.AppCountLogsResponse, error)
	infoResponder   func(*pb.FunctionCallGetInfoRequest, int) (*pb.FunctionCallGetInfoResponse, error)
	streamResponder func(context.Context, *pb.AppGetLogsRequest, int) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error)
}

func (m *mockLogsClient) AppFetchLogs(
	ctx context.Context,
	request *pb.AppFetchLogsRequest,
	opts ...grpc.CallOption,
) (*pb.AppFetchLogsResponse, error) {
	m.mu.Lock()
	m.requests = append(m.requests, request)
	callIndex := len(m.requests) - 1
	m.mu.Unlock()
	return m.responder(ctx, request, callIndex)
}

func (m *mockLogsClient) AppCountLogs(
	ctx context.Context,
	request *pb.AppCountLogsRequest,
	opts ...grpc.CallOption,
) (*pb.AppCountLogsResponse, error) {
	m.mu.Lock()
	m.countRequests = append(m.countRequests, request)
	callIndex := len(m.countRequests) - 1
	m.mu.Unlock()
	return m.countResponder(request, callIndex)
}

func (m *mockLogsClient) FunctionCallGetInfo(
	ctx context.Context,
	request *pb.FunctionCallGetInfoRequest,
	opts ...grpc.CallOption,
) (*pb.FunctionCallGetInfoResponse, error) {
	m.mu.Lock()
	m.infoRequests = append(m.infoRequests, request)
	callIndex := len(m.infoRequests) - 1
	m.mu.Unlock()
	return m.infoResponder(request, callIndex)
}

func (m *mockLogsClient) AppGetLogs(
	ctx context.Context,
	request *pb.AppGetLogsRequest,
	opts ...grpc.CallOption,
) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
	m.mu.Lock()
	m.streamRequests = append(m.streamRequests, request)
	callIndex := len(m.streamRequests) - 1
	m.mu.Unlock()
	return m.streamResponder(ctx, request, callIndex)
}

type mockLogStream struct {
	grpc.ClientStream
	recv func() (*pb.TaskLogsBatch, error)
}

func (m *mockLogStream) Recv() (*pb.TaskLogsBatch, error) {
	return m.recv()
}

func logStream(
	ctx context.Context,
	receives ...logStreamReceive,
) grpc.ServerStreamingClient[pb.TaskLogsBatch] {
	index := 0
	return &mockLogStream{
		recv: func() (*pb.TaskLogsBatch, error) {
			if index < len(receives) {
				receive := receives[index]
				index++
				return receive.batch, receive.err
			}
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}
}

func newLogsTestClient(mockClient pb.ModalClientClient) *Client {
	return &Client{
		cpClient: &clientWithConn{ModalClientClient: mockClient},
	}
}

func collectLogEntries(sequence iter.Seq2[LogEntry, error]) ([]LogEntry, error) {
	var entries []LogEntry
	for entry, err := range sequence {
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func logBatch(prefix string, count int) *pb.TaskLogsBatch {
	items := make([]*pb.TaskLogs, count)
	for index := range count {
		items[index] = pb.TaskLogs_builder{
			Data:           fmt.Sprintf("%s-%03d", prefix, index),
			FileDescriptor: pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT,
		}.Build()
	}
	return pb.TaskLogsBatch_builder{Items: items}.Build()
}

func logBucket(start time.Time, count uint64) *pb.AppCountLogsResponse_LogBucket {
	return pb.AppCountLogsResponse_LogBucket_builder{
		BucketStartAt: timestamppb.New(start),
		StdoutLogs:    count,
	}.Build()
}

func TestFunctionLogsStreamBuildsRequestAndYieldsEntries(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	batch := pb.TaskLogsBatch_builder{
		EntryId: "entry-1",
		AppDone: true,
		Items: []*pb.TaskLogs{
			pb.TaskLogs_builder{Data: ""}.Build(),
			pb.TaskLogs_builder{
				Data:           "hello",
				FileDescriptor: pb.FileDescriptor_FILE_DESCRIPTOR_STDERR,
			}.Build(),
		},
	}.Build()
	mockClient := &mockLogsClient{
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			return logStream(ctx, logStreamReceive{batch: batch}), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	sequence, err := manager.Stream(t.Context(), nil)

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(1))
	g.Expect(entries[0].Source).To(Equal(LogSourceStderr))
	g.Expect(entries[0].Message).To(Equal("hello"))
	g.Expect(entries[0].ObjectID).To(Equal("fu-123"))
	g.Expect(mockClient.streamRequests).To(HaveLen(1))
	request := mockClient.streamRequests[0]
	g.Expect(request.GetAppId()).To(Equal("ap-123"))
	g.Expect(request.GetFunctionId()).To(Equal("fu-123"))
	g.Expect(request.GetFunctionCallId()).To(BeEmpty())
	g.Expect(request.GetLastEntryId()).To(BeEmpty())
	g.Expect(request.GetTimeout()).To(Equal(float32(logStreamRPCTimeout.Seconds())))
}

func TestFunctionLogsStreamReconnectsAfterEOF(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	mockClient := &mockLogsClient{
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			if callIndex == 0 {
				return logStream(
					ctx,
					logStreamReceive{
						batch: pb.TaskLogsBatch_builder{
							EntryId: "entry-1",
							Items:   logBatch("first", 1).GetItems(),
						}.Build(),
					},
					logStreamReceive{err: io.EOF},
				), nil
			}
			return logStream(
				ctx,
				logStreamReceive{
					batch: pb.TaskLogsBatch_builder{
						EntryId: "entry-2",
						AppDone: true,
						Items:   logBatch("second", 1).GetItems(),
					}.Build(),
				},
			), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	sequence, err := manager.Stream(t.Context(), nil)

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(2))
	g.Expect(entries[0].Message).To(Equal("first-000"))
	g.Expect(entries[1].Message).To(Equal("second-000"))
	g.Expect(mockClient.streamRequests).To(HaveLen(2))
	g.Expect(mockClient.streamRequests[1].GetLastEntryId()).To(Equal("entry-1"))
}

func TestFunctionLogsStreamStopsAfterIdleTimeout(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	mockClient := &mockLogsClient{
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			return logStream(ctx), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	timeout := 50 * time.Millisecond

	params := &LogStreamParams{Timeout: &timeout}
	sequence, err := manager.Stream(
		t.Context(),
		params,
	)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params).To(Equal(&LogStreamParams{Timeout: &timeout}))
	timeout = 0
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(BeEmpty())
	g.Expect(mockClient.streamRequests).To(HaveLen(1))
}

func TestFunctionLogsStreamIdleTimeoutAppliesDuringRetries(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	mockClient := &mockLogsClient{
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			return logStream(
				ctx,
				logStreamReceive{err: status.Error(codes.Unavailable, "try again")},
			), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	timeout := 50 * time.Millisecond

	sequence, err := manager.Stream(
		t.Context(),
		&LogStreamParams{Timeout: &timeout},
	)

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(BeEmpty())
	g.Expect(len(mockClient.streamRequests)).To(BeNumerically(">=", 2))
}

func TestFunctionLogsStreamDoesNotStartWithNonpositiveTimeout(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	mockClient := &mockLogsClient{}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	timeout := time.Duration(0)

	sequence, err := manager.Stream(
		t.Context(),
		&LogStreamParams{Timeout: &timeout},
	)

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(BeEmpty())
	g.Expect(mockClient.streamRequests).To(BeEmpty())
}

func TestFunctionLogsStreamYieldsContextCancellation(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	streamStarted := make(chan struct{})
	mockClient := &mockLogsClient{
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			close(streamStarted)
			return logStream(ctx), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	ctx, cancel := context.WithCancel(t.Context())
	sequence, err := manager.Stream(ctx, nil)
	g.Expect(err).NotTo(HaveOccurred())
	result := make(chan error, 1)
	go func() {
		_, err := collectLogEntries(sequence)
		result <- err
	}()

	<-streamStarted
	cancel()

	select {
	case err := <-result:
		g.Expect(err).To(MatchError(context.Canceled))
	case <-time.After(time.Second):
		t.Fatal("stream did not stop after context cancellation")
	}
}

func TestFunctionLogsStreamCancelsRPCWhenIterationStops(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	rpcCanceled := make(chan struct{})
	var cancelOnce sync.Once
	mockClient := &mockLogsClient{
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			sent := false
			return &mockLogStream{
				recv: func() (*pb.TaskLogsBatch, error) {
					if !sent {
						sent = true
						return logBatch("entry", 1), nil
					}
					<-ctx.Done()
					cancelOnce.Do(func() { close(rpcCanceled) })
					return nil, ctx.Err()
				},
			}, nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	sequence, err := manager.Stream(t.Context(), nil)
	g.Expect(err).NotTo(HaveOccurred())

	for range sequence {
		break
	}

	select {
	case <-rpcCanceled:
	case <-time.After(time.Second):
		t.Fatal("active AppGetLogs RPC was not canceled")
	}
	g.Expect(mockClient.streamRequests).To(HaveLen(1))
}

func TestFunctionCallLogsStreamDrainsAfterCompletion(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	streamStarted := make(chan struct{})
	var streamStartedOnce sync.Once
	mockClient := &mockLogsClient{
		infoResponder: func(
			request *pb.FunctionCallGetInfoRequest,
			callIndex int,
		) (*pb.FunctionCallGetInfoResponse, error) {
			<-streamStarted
			return pb.FunctionCallGetInfoResponse_builder{
				Info: pb.FunctionCallInfo_builder{
					TotalInputs:     1,
					SucceededInputs: pb.InputCategoryInfo_builder{Total: 1}.Build(),
				}.Build(),
			}.Build(), nil
		},
		streamResponder: func(
			ctx context.Context,
			request *pb.AppGetLogsRequest,
			callIndex int,
		) (grpc.ServerStreamingClient[pb.TaskLogsBatch], error) {
			if callIndex == 0 {
				streamStartedOnce.Do(func() { close(streamStarted) })
				return logStream(ctx), nil
			}
			return logStream(
				ctx,
				logStreamReceive{batch: logBatch("drained", 1)},
				logStreamReceive{err: io.EOF},
			), nil
		},
	}
	manager := &FunctionCallLogsManager{
		client:         newLogsTestClient(mockClient),
		appID:          "ap-123",
		functionID:     "fu-123",
		functionCallID: "fc-123",
	}

	sequence, err := manager.Stream(t.Context(), nil)

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(1))
	g.Expect(entries[0].Message).To(Equal("drained-000"))
	g.Expect(entries[0].ObjectID).To(Equal("fc-123"))
	g.Expect(mockClient.infoRequests).To(HaveLen(1))
	g.Expect(mockClient.infoRequests[0].GetFunctionId()).To(Equal("fu-123"))
	g.Expect(mockClient.infoRequests[0].GetFunctionCallId()).To(Equal("fc-123"))
	g.Expect(mockClient.streamRequests).To(HaveLen(2))
	g.Expect(mockClient.streamRequests[0].GetTimeout()).To(Equal(float32(logStreamRPCTimeout.Seconds())))
	drainRequest := mockClient.streamRequests[1]
	g.Expect(drainRequest.GetTimeout()).To(Equal(float32(logStreamDrainTimeout.Seconds())))
	g.Expect(drainRequest.GetFunctionId()).To(Equal("fu-123"))
	g.Expect(drainRequest.GetFunctionCallId()).To(Equal("fc-123"))
}

func TestFunctionCallCompleteKeepsWaitingWhenCallNotFound(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	mockClient := &mockLogsClient{
		infoResponder: func(
			request *pb.FunctionCallGetInfoRequest,
			callIndex int,
		) (*pb.FunctionCallGetInfoResponse, error) {
			return nil, status.Error(codes.NotFound, "function call not found")
		},
	}
	manager := &FunctionCallLogsManager{
		client:         newLogsTestClient(mockClient),
		appID:          "ap-123",
		functionID:     "fu-123",
		functionCallID: "fc-123",
	}

	complete, err := manager.functionCallComplete(t.Context())

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(complete).To(BeFalse())
	g.Expect(mockClient.infoRequests).To(HaveLen(functionCallGetInfoMaxAttempts))
}

func TestFunctionLogsTailDefaultsToOneHundredEntries(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	allLogs := logBatch("entry", 150).GetItems()
	mockClient := &mockLogsClient{
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			limit := int(request.GetLimit())
			return pb.AppFetchLogsResponse_builder{
				Batches: []*pb.TaskLogsBatch{
					pb.TaskLogsBatch_builder{Items: allLogs[len(allLogs)-limit:]}.Build(),
				},
			}.Build(), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	params := &LogTailParams{Source: LogSourceStdout}
	sequence, err := manager.Tail(t.Context(), params)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params).To(Equal(&LogTailParams{Source: LogSourceStdout}))
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(defaultLogTailEntries))
	g.Expect(entries[0].Message).To(Equal("entry-050"))
	g.Expect(entries[len(entries)-1].Message).To(Equal("entry-149"))
	g.Expect(mockClient.requests).To(HaveLen(1))

	request := mockClient.requests[0]
	g.Expect(request.GetLimit()).To(Equal(uint32(defaultLogTailEntries)))
	g.Expect(request.GetSource()).To(Equal(pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT))
	g.Expect(request.GetAppId()).To(Equal("ap-123"))
	g.Expect(request.GetFunctionId()).To(Equal("fu-123"))
	g.Expect(request.GetUntil().AsTime().Sub(request.GetSince().AsTime())).To(Equal(time.Hour))
}

func TestFunctionLogsTailProgressivelyWidensLookback(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	counts := []int{20, 70, 100}
	mockClient := &mockLogsClient{
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			return pb.AppFetchLogsResponse_builder{
				Batches: []*pb.TaskLogsBatch{logBatch(fmt.Sprintf("window-%d", callIndex), counts[callIndex])},
			}.Build(), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	sequence, err := manager.Tail(t.Context(), nil)

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(defaultLogTailEntries))
	g.Expect(entries[0].Message).To(Equal("window-2-000"))
	g.Expect(entries[len(entries)-1].Message).To(Equal("window-2-099"))
	g.Expect(mockClient.requests).To(HaveLen(3))

	for index, request := range mockClient.requests {
		g.Expect(request.GetUntil().AsTime().Sub(request.GetSince().AsTime())).To(Equal(logTailLookbacks[index]))
		g.Expect(request.GetUntil()).To(Equal(mockClient.requests[0].GetUntil()))
	}
}

func TestFunctionCallLogsTailFiltersAndBuildsEntries(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	timestamp := time.Date(2026, time.July, 28, 12, 34, 56, 789, time.UTC)
	mockClient := &mockLogsClient{
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			item := pb.TaskLogs_builder{
				Data:           "hello",
				FileDescriptor: pb.FileDescriptor_FILE_DESCRIPTOR_STDERR,
				FunctionCallId: "fc-123",
				Timestamp:      1,
				TimestampNs:    uint64(timestamp.UnixNano()),
			}.Build()
			batch := pb.TaskLogsBatch_builder{
				TaskId:  "ta-123",
				InputId: "in-123",
				Items:   []*pb.TaskLogs{item},
			}.Build()
			return pb.AppFetchLogsResponse_builder{Batches: []*pb.TaskLogsBatch{batch}}.Build(), nil
		},
	}
	manager := &FunctionCallLogsManager{
		client:         newLogsTestClient(mockClient),
		appID:          "ap-123",
		functionID:     "fu-123",
		functionCallID: "fc-123",
	}

	params := &LogTailParams{Source: LogSourceStderr}
	sequence, err := manager.Tail(t.Context(), params)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params).To(Equal(&LogTailParams{Source: LogSourceStderr}))
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(Equal([]LogEntry{{
		Timestamp:  timestamp,
		Source:     LogSourceStderr,
		Message:    "hello",
		ObjectID:   "fc-123",
		ContextIDs: []string{"in-123", "ta-123"},
	}}))

	request := mockClient.requests[0]
	g.Expect(request.GetLimit()).To(Equal(uint32(defaultLogTailEntries)))
	g.Expect(request.GetFunctionId()).To(Equal("fu-123"))
	g.Expect(request.GetFunctionCallId()).To(Equal("fc-123"))
	g.Expect(request.GetSource()).To(Equal(pb.FileDescriptor_FILE_DESCRIPTOR_STDERR))
}

func TestLogsTailValidatesParams(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name   string
		params *LogTailParams
	}{
		{name: "negative entries", params: &LogTailParams{Entries: -1}},
		{name: "too many entries", params: &LogTailParams{Entries: maxLogFetchEntries + 1}},
		{name: "invalid source", params: &LogTailParams{Source: LogSource("other")}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)

			mockClient := &mockLogsClient{
				responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
					return nil, errors.New("RPC should not be called")
				},
			}
			manager := &FunctionLogsManager{
				client:     newLogsTestClient(mockClient),
				appID:      "ap-123",
				functionID: "fu-123",
			}

			sequence, err := manager.Tail(t.Context(), testCase.params)

			g.Expect(err).To(BeAssignableToTypeOf(InvalidError{}))
			g.Expect(sequence).To(BeNil())
			g.Expect(mockClient.requests).To(BeEmpty())
		})
	}
}

func TestLogsTailReturnsFetchError(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	fetchError := errors.New("backend unavailable")
	mockClient := &mockLogsClient{
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			return nil, fetchError
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	sequence, err := manager.Tail(t.Context(), nil)

	g.Expect(err).NotTo(HaveOccurred())
	_, err = collectLogEntries(sequence)
	g.Expect(err).To(MatchError(ContainSubstring("AppFetchLogs failed")))
	g.Expect(errors.Is(err, fetchError)).To(BeTrue())
	g.Expect(mockClient.requests).To(HaveLen(1))
}

func TestBuildLogFetchIntervals(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	start := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	ranges := []logRange{
		{start: start, end: start.Add(time.Minute), count: 1_000},
		{start: start.Add(time.Minute), end: start.Add(2 * time.Minute), count: 1_000},
		{start: start.Add(2 * time.Minute), end: start.Add(3 * time.Minute), count: 0},
		{start: start.Add(3 * time.Minute), end: start.Add(4 * time.Minute), count: 19_000},
		{start: start.Add(4 * time.Minute), end: start.Add(5 * time.Minute), count: 2_000},
	}

	g.Expect(buildLogFetchIntervals(ranges)).To(Equal([]logInterval{
		{since: start, until: start.Add(2 * time.Minute)},
		{since: start.Add(3 * time.Minute), until: start.Add(4 * time.Minute)},
		{since: start.Add(4 * time.Minute), until: start.Add(5 * time.Minute)},
	}))
}

func TestRunLogRefinementsClampsFirstSubrangeToParent(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	start := time.Unix(0, 0).UTC()
	parent := logRange{
		start: start.Add(6 * time.Second),
		end:   start.Add(12 * time.Second),
		count: maxLogFetchEntries + 1,
	}
	mockClient := &mockLogsClient{
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			return pb.AppCountLogsResponse_builder{
				Buckets: []*pb.AppCountLogsResponse_LogBucket{
					logBucket(start.Add(4*time.Second), 100),
					logBucket(start.Add(8*time.Second), 100),
				},
			}.Build(), nil
		},
	}
	client := newLogsTestClient(mockClient)

	results := runLogRefinements(
		t.Context(),
		client,
		"ap-123",
		[]logRange{parent},
		[]logRefinement{{index: 0, bucketSize: 4 * time.Second}},
		logsFilters{},
	)

	g.Expect(results).To(Equal([]logRefinementResult{{
		ranges: []logRange{
			{start: start.Add(6 * time.Second), end: start.Add(8 * time.Second), count: 100},
			{start: start.Add(8 * time.Second), end: start.Add(12 * time.Second), count: 100},
		},
	}}))
	g.Expect(mockClient.countRequests).To(HaveLen(1))
	g.Expect(mockClient.countRequests[0].GetSince().AsTime()).To(Equal(parent.start))
	g.Expect(mockClient.countRequests[0].GetUntil().AsTime()).To(Equal(parent.end))
	g.Expect(mockClient.countRequests[0].GetBucketSecs()).To(Equal(uint32(4)))
}

func TestFunctionLogsFetch(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	since := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	until := since.Add(10 * time.Minute)
	mockClient := &mockLogsClient{
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			return pb.AppCountLogsResponse_builder{
				Buckets: []*pb.AppCountLogsResponse_LogBucket{logBucket(since, 1)},
			}.Build(), nil
		},
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			return pb.AppFetchLogsResponse_builder{
				Batches: []*pb.TaskLogsBatch{logBatch("fetch", 1)},
			}.Build(), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	params := &FunctionLogFetchParams{
		Until:      &until,
		Source:     LogSourceStderr,
		SearchText: "needle",
	}
	sequence, err := manager.Fetch(t.Context(), since, params)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params).To(Equal(&FunctionLogFetchParams{
		Until:      &until,
		Source:     LogSourceStderr,
		SearchText: "needle",
	}))
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(1))
	g.Expect(entries[0].ObjectID).To(Equal("fu-123"))
	g.Expect(mockClient.countRequests).To(HaveLen(1))
	g.Expect(mockClient.requests).To(HaveLen(1))

	countRequest := mockClient.countRequests[0]
	g.Expect(countRequest.GetSince().AsTime()).To(Equal(since))
	g.Expect(countRequest.GetUntil().AsTime()).To(Equal(until))
	g.Expect(countRequest.GetBucketSecs()).To(Equal(uint32(6)))
	g.Expect(countRequest.GetSource()).To(Equal(pb.FileDescriptor_FILE_DESCRIPTOR_STDERR))
	g.Expect(countRequest.GetFunctionId()).To(Equal("fu-123"))
	g.Expect(countRequest.GetSearchText()).To(Equal("needle"))

	fetchRequest := mockClient.requests[0]
	g.Expect(fetchRequest.GetLimit()).To(Equal(uint32(maxLogFetchEntries)))
	g.Expect(fetchRequest.GetSince().AsTime()).To(Equal(since))
	g.Expect(fetchRequest.GetUntil().AsTime()).To(Equal(since.Add(6 * time.Second)))
	g.Expect(fetchRequest.GetSource()).To(Equal(pb.FileDescriptor_FILE_DESCRIPTOR_STDERR))
	g.Expect(fetchRequest.GetSearchText()).To(Equal("needle"))
}

func TestFunctionCallLogsFetchDefaultsSinceToCreationTime(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	createdAt := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	until := createdAt.Add(time.Minute)
	mockClient := &mockLogsClient{
		infoResponder: func(request *pb.FunctionCallGetInfoRequest, callIndex int) (*pb.FunctionCallGetInfoResponse, error) {
			return pb.FunctionCallGetInfoResponse_builder{
				Info: pb.FunctionCallInfo_builder{CreatedAt: float64(createdAt.Unix())}.Build(),
			}.Build(), nil
		},
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			return pb.AppCountLogsResponse_builder{
				Buckets: []*pb.AppCountLogsResponse_LogBucket{logBucket(createdAt, 1)},
			}.Build(), nil
		},
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			return pb.AppFetchLogsResponse_builder{
				Batches: []*pb.TaskLogsBatch{logBatch("fetch", 1)},
			}.Build(), nil
		},
	}
	manager := &FunctionCallLogsManager{
		client:         newLogsTestClient(mockClient),
		appID:          "ap-123",
		functionID:     "fu-123",
		functionCallID: "fc-123",
	}

	params := &FunctionCallLogFetchParams{Until: &until}
	sequence, err := manager.Fetch(t.Context(), params)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params).To(Equal(&FunctionCallLogFetchParams{Until: &until}))
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(1))
	g.Expect(entries[0].ObjectID).To(Equal("fc-123"))
	g.Expect(mockClient.infoRequests).To(HaveLen(1))
	g.Expect(mockClient.infoRequests[0].GetFunctionId()).To(Equal("fu-123"))
	g.Expect(mockClient.infoRequests[0].GetFunctionCallId()).To(Equal("fc-123"))
	g.Expect(mockClient.countRequests[0].GetSince().AsTime()).To(Equal(createdAt))
	g.Expect(mockClient.countRequests[0].GetFunctionCallId()).To(Equal("fc-123"))
}

func TestLogsFetchValidatesParamsBeforeReturningSequence(t *testing.T) {
	t.Parallel()

	since := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	testCases := []struct {
		name  string
		fetch func(context.Context, *mockLogsClient) (iter.Seq2[LogEntry, error], error)
	}{
		{
			name: "function invalid source",
			fetch: func(ctx context.Context, mockClient *mockLogsClient) (iter.Seq2[LogEntry, error], error) {
				manager := &FunctionLogsManager{
					client:     newLogsTestClient(mockClient),
					appID:      "ap-123",
					functionID: "fu-123",
				}
				return manager.Fetch(ctx, since, &FunctionLogFetchParams{Source: LogSource("other")})
			},
		},
		{
			name: "function range too wide",
			fetch: func(ctx context.Context, mockClient *mockLogsClient) (iter.Seq2[LogEntry, error], error) {
				manager := &FunctionLogsManager{
					client:     newLogsTestClient(mockClient),
					appID:      "ap-123",
					functionID: "fu-123",
				}
				until := since.Add(maxLogFetchRange + time.Second)
				return manager.Fetch(ctx, since, &FunctionLogFetchParams{Until: &until})
			},
		},
		{
			name: "function call invalid source",
			fetch: func(ctx context.Context, mockClient *mockLogsClient) (iter.Seq2[LogEntry, error], error) {
				manager := &FunctionCallLogsManager{
					client:         newLogsTestClient(mockClient),
					appID:          "ap-123",
					functionID:     "fu-123",
					functionCallID: "fc-123",
				}
				return manager.Fetch(ctx, &FunctionCallLogFetchParams{Source: LogSource("other")})
			},
		},
		{
			name: "function call range too wide",
			fetch: func(ctx context.Context, mockClient *mockLogsClient) (iter.Seq2[LogEntry, error], error) {
				manager := &FunctionCallLogsManager{
					client:         newLogsTestClient(mockClient),
					appID:          "ap-123",
					functionID:     "fu-123",
					functionCallID: "fc-123",
				}
				until := since.Add(maxLogFetchRange + time.Second)
				return manager.Fetch(ctx, &FunctionCallLogFetchParams{Since: &since, Until: &until})
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			mockClient := &mockLogsClient{}

			sequence, err := testCase.fetch(t.Context(), mockClient)

			g.Expect(err).To(BeAssignableToTypeOf(InvalidError{}))
			g.Expect(sequence).To(BeNil())
			g.Expect(mockClient.infoRequests).To(BeEmpty())
			g.Expect(mockClient.countRequests).To(BeEmpty())
			g.Expect(mockClient.requests).To(BeEmpty())
		})
	}
}

func TestFunctionLogsFetchRefinesDenseRanges(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	since := time.Date(2026, time.July, 28, 0, 0, 0, 0, time.UTC)
	until := since.Add(24 * time.Hour)
	mockClient := &mockLogsClient{
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			switch request.GetBucketSecs() {
			case 900:
				return pb.AppCountLogsResponse_builder{
					Buckets: []*pb.AppCountLogsResponse_LogBucket{
						logBucket(since, maxLogFetchEntries+1),
					},
				}.Build(), nil
			case 720:
				return pb.AppCountLogsResponse_builder{
					Buckets: []*pb.AppCountLogsResponse_LogBucket{
						logBucket(since, 10_000),
						logBucket(since.Add(12*time.Minute), 10_000),
					},
				}.Build(), nil
			default:
				return nil, fmt.Errorf("unexpected bucket size: %d", request.GetBucketSecs())
			}
		},
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			return pb.AppFetchLogsResponse_builder{
				Batches: []*pb.TaskLogsBatch{logBatch(
					fmt.Sprintf("%d", int(request.GetSince().AsTime().Sub(since)/time.Minute)),
					1,
				)},
			}.Build(), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	sequence, err := manager.Fetch(t.Context(), since, &FunctionLogFetchParams{Until: &until})

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(2))
	g.Expect(mockClient.countRequests).To(HaveLen(2))
	g.Expect(mockClient.countRequests[0].GetBucketSecs()).To(Equal(uint32(900)))
	g.Expect(mockClient.countRequests[1].GetBucketSecs()).To(Equal(uint32(720)))
	g.Expect(mockClient.requests).To(HaveLen(2))
	requests := slices.Clone(mockClient.requests)
	slices.SortFunc(requests, func(left *pb.AppFetchLogsRequest, right *pb.AppFetchLogsRequest) int {
		return left.GetSince().AsTime().Compare(right.GetSince().AsTime())
	})
	g.Expect(requests[0].GetUntil().AsTime()).To(Equal(since.Add(12 * time.Minute)))
	g.Expect(requests[1].GetSince().AsTime()).To(Equal(since.Add(12 * time.Minute)))
	g.Expect(requests[1].GetUntil().AsTime()).To(Equal(since.Add(15 * time.Minute)))
}

func TestFunctionLogsFetchLimitsConcurrencyAndYieldsInOrder(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	since := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	until := since.Add(24 * time.Second)
	buckets := make([]*pb.AppCountLogsResponse_LogBucket, 12)
	for index := range buckets {
		buckets[index] = logBucket(since.Add(time.Duration(index)*2*time.Second), logIntervalEntryThreshold)
	}

	var active atomic.Int32
	var maximumActive atomic.Int32
	var started atomic.Int32
	release := make(chan struct{})
	mockClient := &mockLogsClient{
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			return pb.AppCountLogsResponse_builder{Buckets: buckets}.Build(), nil
		},
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			current := active.Add(1)
			defer active.Add(-1)
			for {
				maximum := maximumActive.Load()
				if current <= maximum || maximumActive.CompareAndSwap(maximum, current) {
					break
				}
			}
			if started.Add(1) == maxConcurrentLogFetches {
				close(release)
			}
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}

			index := int(request.GetSince().AsTime().Sub(since) / (2 * time.Second))
			return pb.AppFetchLogsResponse_builder{
				Batches: []*pb.TaskLogsBatch{logBatch(fmt.Sprintf("interval-%02d", index), 1)},
			}.Build(), nil
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}

	sequence, err := manager.Fetch(t.Context(), since, &FunctionLogFetchParams{Until: &until})

	g.Expect(err).NotTo(HaveOccurred())
	entries, err := collectLogEntries(sequence)

	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(entries).To(HaveLen(12))
	for index, entry := range entries {
		g.Expect(entry.Message).To(Equal(fmt.Sprintf("interval-%02d-000", index)))
	}
	g.Expect(maximumActive.Load()).To(Equal(int32(maxConcurrentLogFetches)))
	g.Expect(active.Load()).To(BeZero())
}

func TestFunctionLogsFetchObservesErrorsInIntervalOrder(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	since := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	until := since.Add(6 * time.Second)
	buckets := []*pb.AppCountLogsResponse_LogBucket{
		logBucket(since, logIntervalEntryThreshold),
		logBucket(since.Add(2*time.Second), logIntervalEntryThreshold),
		logBucket(since.Add(4*time.Second), logIntervalEntryThreshold),
	}

	releaseFirst := make(chan struct{})
	laterFailed := make(chan struct{})
	var firstWasCancelled atomic.Bool
	laterError := errors.New("later interval failed")
	mockClient := &mockLogsClient{
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			return pb.AppCountLogsResponse_builder{Buckets: buckets}.Build(), nil
		},
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			index := int(request.GetSince().AsTime().Sub(since) / (2 * time.Second))
			switch index {
			case 0:
				select {
				case <-releaseFirst:
					return pb.AppFetchLogsResponse_builder{
						Batches: []*pb.TaskLogsBatch{logBatch("first", 1)},
					}.Build(), nil
				case <-ctx.Done():
					firstWasCancelled.Store(true)
					return nil, ctx.Err()
				}
			case 1:
				close(laterFailed)
				return nil, laterError
			default:
				return pb.AppFetchLogsResponse_builder{
					Batches: []*pb.TaskLogsBatch{logBatch("third", 1)},
				}.Build(), nil
			}
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	sequence, err := manager.Fetch(t.Context(), since, &FunctionLogFetchParams{Until: &until})
	g.Expect(err).NotTo(HaveOccurred())

	result := make(chan error, 1)
	go func() {
		_, err := collectLogEntries(sequence)
		result <- err
	}()

	<-laterFailed
	g.Expect(firstWasCancelled.Load()).To(BeFalse())
	close(releaseFirst)

	err = <-result
	g.Expect(errors.Is(err, laterError)).To(BeTrue())
	g.Expect(firstWasCancelled.Load()).To(BeFalse())
}

func TestFunctionLogsFetchCancelsRequestsWhenIterationStops(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)

	since := time.Date(2026, time.July, 28, 12, 0, 0, 0, time.UTC)
	until := since.Add(24 * time.Second)
	buckets := make([]*pb.AppCountLogsResponse_LogBucket, 12)
	for index := range buckets {
		buckets[index] = logBucket(since.Add(time.Duration(index)*2*time.Second), logIntervalEntryThreshold)
	}

	var active atomic.Int32
	var cancelled atomic.Int32
	mockClient := &mockLogsClient{
		countResponder: func(request *pb.AppCountLogsRequest, callIndex int) (*pb.AppCountLogsResponse, error) {
			return pb.AppCountLogsResponse_builder{Buckets: buckets}.Build(), nil
		},
		responder: func(ctx context.Context, request *pb.AppFetchLogsRequest, callIndex int) (*pb.AppFetchLogsResponse, error) {
			active.Add(1)
			defer active.Add(-1)

			index := int(request.GetSince().AsTime().Sub(since) / (2 * time.Second))
			if index == 0 {
				return pb.AppFetchLogsResponse_builder{
					Batches: []*pb.TaskLogsBatch{logBatch("first", 1)},
				}.Build(), nil
			}

			<-ctx.Done()
			cancelled.Add(1)
			return nil, ctx.Err()
		},
	}
	manager := &FunctionLogsManager{
		client:     newLogsTestClient(mockClient),
		appID:      "ap-123",
		functionID: "fu-123",
	}
	sequence, err := manager.Fetch(t.Context(), since, &FunctionLogFetchParams{Until: &until})
	g.Expect(err).NotTo(HaveOccurred())

	yielded := 0
	for _, err := range sequence {
		g.Expect(err).NotTo(HaveOccurred())
		yielded++
		break
	}

	g.Expect(yielded).To(Equal(1))
	g.Expect(active.Load()).To(BeZero())
	g.Expect(cancelled.Load()).To(BeNumerically(">", 0))
}
