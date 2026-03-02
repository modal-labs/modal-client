package test

import (
	"errors"
	"strconv"
	"sync"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestQueueInvalidName(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	tc := newTestClient(t)

	for _, name := range []string{"has space", "has/slash", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"} {
		_, err := tc.Queues.FromName(t.Context(), name, nil)
		g.Expect(err).Should(gomega.HaveOccurred())
	}
}

func TestQueueEphemeral(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	queue, err := tc.Queues.Ephemeral(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer queue.CloseEphemeral()
	g.Expect(queue.Name).To(gomega.BeEmpty())

	err = queue.Put(ctx, 123, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	len, err := queue.Len(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(len).To(gomega.Equal(1))

	result, err := queue.Get(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.Equal(int64(123)))
}

func TestQueueSuite1(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	queue, err := tc.Queues.Ephemeral(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer queue.CloseEphemeral()

	n, err := queue.Len(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(0))

	g.Expect(queue.Put(ctx, 123, nil)).ToNot(gomega.HaveOccurred())

	n, err = queue.Len(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(1))

	item, err := queue.Get(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(item).To(gomega.Equal(int64(123)))

	// put, then non-blocking get
	g.Expect(queue.Put(ctx, 432, nil)).ToNot(gomega.HaveOccurred())

	var timeout time.Duration
	item, err = queue.Get(ctx, &modal.QueueGetParams{Timeout: &timeout})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(item).To(gomega.Equal(int64(432)))

	// queue is now empty – non-blocking get should error
	_, err = queue.Get(ctx, &modal.QueueGetParams{Timeout: &timeout})
	g.Expect(errors.As(err, &modal.QueueEmptyError{})).To(gomega.BeTrue())

	n, err = queue.Len(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(0))

	g.Expect(queue.PutMany(ctx, []any{1, 2, 3}, nil)).ToNot(gomega.HaveOccurred())

	results := make([]int64, 0, 3)
	for v, err := range queue.Iterate(ctx, nil) {
		g.Expect(err).ToNot(gomega.HaveOccurred())
		results = append(results, v.(int64))
	}
	g.Expect(results).To(gomega.Equal([]int64{1, 2, 3}))
}

func TestQueueSuite2(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	queue, err := tc.Queues.Ephemeral(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer queue.CloseEphemeral()

	var wg sync.WaitGroup
	results := make([]int64, 0, 10)

	// producer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 10 {
			_ = queue.Put(ctx, i, nil) // ignore error for brevity
		}
	}()

	// consumer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for v, err := range queue.Iterate(ctx, &modal.QueueIterateParams{ItemPollTimeout: time.Second}) {
			g.Expect(err).ToNot(gomega.HaveOccurred())
			results = append(results, v.(int64))
		}
	}()

	wg.Wait()
	g.Expect(results).To(gomega.Equal([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
}

func TestQueuePutAndGetMany(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	queue, err := tc.Queues.Ephemeral(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer queue.CloseEphemeral()

	g.Expect(queue.PutMany(ctx, []any{1, 2, 3}, nil)).ToNot(gomega.HaveOccurred())

	n, err := queue.Len(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(3))

	items, err := queue.GetMany(ctx, 3, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(items).To(gomega.Equal([]any{int64(1), int64(2), int64(3)}))
}

func TestQueueNonBlocking(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	queue, err := tc.Queues.Ephemeral(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer queue.CloseEphemeral()

	var timeout time.Duration
	err = queue.Put(ctx, 123, &modal.QueuePutParams{Timeout: &timeout})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	n, err := queue.Len(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(n).To(gomega.Equal(1))

	item, err := queue.Get(ctx, &modal.QueueGetParams{Timeout: &timeout})
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(item).To(gomega.Equal(int64(123)))
}

func TestQueueNonEphemeral(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	queueName := "test-queue-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	queue1, err := tc.Queues.FromName(ctx, queueName, &modal.QueueFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(queue1.Name).To(gomega.Equal(queueName))

	defer func() {
		err := tc.Queues.Delete(ctx, queueName, nil)
		g.Expect(err).ShouldNot(gomega.HaveOccurred())

		_, err = tc.Queues.FromName(ctx, queueName, nil)
		g.Expect(err).Should(gomega.HaveOccurred())
	}()

	err = queue1.Put(ctx, "data", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	queue2, err := tc.Queues.FromName(ctx, queueName, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	value, err := queue2.Get(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(value).To(gomega.Equal("data"))
}

func TestQueueDeleteSuccess(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/QueueGetOrCreate",
		func(req *pb.QueueGetOrCreateRequest) (*pb.QueueGetOrCreateResponse, error) {
			return pb.QueueGetOrCreateResponse_builder{
				QueueId: "qu-test-123",
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "/QueueDelete",
		func(req *pb.QueueDeleteRequest) (*emptypb.Empty, error) {
			g.Expect(req.GetQueueId()).To(gomega.Equal("qu-test-123"))
			return &emptypb.Empty{}, nil
		},
	)

	err := mock.Queues.Delete(ctx, "test-queue", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestQueueDeleteWithAllowMissing(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/QueueGetOrCreate",
		func(req *pb.QueueGetOrCreateRequest) (*pb.QueueGetOrCreateResponse, error) {
			return nil, modal.NotFoundError{Exception: "Queue 'missing' not found"}
		},
	)

	err := mock.Queues.Delete(ctx, "missing", &modal.QueueDeleteParams{
		AllowMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestQueueDeleteWithAllowMissingDeleteRPCNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "/QueueGetOrCreate",
		func(req *pb.QueueGetOrCreateRequest) (*pb.QueueGetOrCreateResponse, error) {
			return pb.QueueGetOrCreateResponse_builder{QueueId: "qu-test-123"}.Build(), nil
		},
	)

	grpcmock.HandleUnary(mock, "/QueueDelete",
		func(req *pb.QueueDeleteRequest) (*emptypb.Empty, error) {
			return nil, status.Errorf(codes.NotFound, "Queue not found")
		},
	)

	err := mock.Queues.Delete(ctx, "test-queue", &modal.QueueDeleteParams{AllowMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestQueueDeleteWithAllowMissingFalseThrows(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/QueueGetOrCreate",
		func(req *pb.QueueGetOrCreateRequest) (*pb.QueueGetOrCreateResponse, error) {
			return nil, modal.NotFoundError{Exception: "Queue 'missing' not found"}
		},
	)

	err := mock.Queues.Delete(ctx, "missing", &modal.QueueDeleteParams{
		AllowMissing: false,
	})
	g.Expect(err).Should(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
