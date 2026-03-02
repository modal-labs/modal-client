package modal

// Queue object, to be used with Modal Queues.

import (
	"context"
	"fmt"
	"iter"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// QueueService provides Queue related operations.
type QueueService interface {
	Ephemeral(ctx context.Context, params *QueueEphemeralParams) (*Queue, error)
	FromName(ctx context.Context, name string, params *QueueFromNameParams) (*Queue, error)
	Delete(ctx context.Context, name string, params *QueueDeleteParams) error
}

type queueServiceImpl struct{ client *Client }

const queueInitialPutBackoff = 100 * time.Millisecond
const queueDefaultPartitionTTL = 24 * time.Hour

func validatePartitionKey(partition string) ([]byte, error) {
	if partition == "" {
		return nil, nil // default partition
	}
	b := []byte(partition)
	if len(b) == 0 || len(b) > 64 {
		return nil, InvalidError{"Queue partition key must be 1–64 bytes long"}
	}
	return b, nil
}

// Queue is a distributed, FIFO queue for data flow in Modal Apps.
type Queue struct {
	QueueID         string
	Name            string
	cancelEphemeral context.CancelFunc

	client *Client
}

// QueueEphemeralParams are options for client.Queues.Ephemeral.
type QueueEphemeralParams struct {
	Environment string
}

// Ephemeral creates a nameless, temporary Queue, that persists until CloseEphemeral is called, or the process exits.
func (s *queueServiceImpl) Ephemeral(ctx context.Context, params *QueueEphemeralParams) (*Queue, error) {
	if params == nil {
		params = &QueueEphemeralParams{}
	}

	resp, err := s.client.cpClient.QueueGetOrCreate(ctx, pb.QueueGetOrCreateRequest_builder{
		ObjectCreationType: pb.ObjectCreationType_OBJECT_CREATION_TYPE_EPHEMERAL,
		EnvironmentName:    environmentName(params.Environment, s.client.profile),
	}.Build())
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Created ephemeral Queue", "queue_id", resp.GetQueueId())

	ephemeralCtx, cancel := context.WithCancel(context.Background())
	startEphemeralHeartbeat(ephemeralCtx, func() error {
		_, err := s.client.cpClient.QueueHeartbeat(ephemeralCtx, pb.QueueHeartbeatRequest_builder{
			QueueId: resp.GetQueueId(),
		}.Build())
		return err
	})

	q := &Queue{
		QueueID:         resp.GetQueueId(),
		cancelEphemeral: cancel,
		client:          s.client,
	}

	return q, nil
}

// CloseEphemeral deletes an ephemeral Queue, only used with QueueEphemeral.
func (q *Queue) CloseEphemeral() {
	if q.cancelEphemeral != nil {
		q.cancelEphemeral()
	} else {
		// We panic in this case because of invalid usage. In general, methods
		// used with `defer` like CloseEphemeral should not return errors.
		panic(fmt.Sprintf("Queue %s is not ephemeral", q.QueueID))
	}
}

// QueueFromNameParams are options for client.Queues.FromName.
type QueueFromNameParams struct {
	Environment     string
	CreateIfMissing bool
}

// FromName references a named Queue, creating if necessary.
func (s *queueServiceImpl) FromName(ctx context.Context, name string, params *QueueFromNameParams) (*Queue, error) {
	if params == nil {
		params = &QueueFromNameParams{}
	}

	creationType := pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED
	if params.CreateIfMissing {
		creationType = pb.ObjectCreationType_OBJECT_CREATION_TYPE_CREATE_IF_MISSING
	}

	resp, err := s.client.cpClient.QueueGetOrCreate(ctx, pb.QueueGetOrCreateRequest_builder{
		DeploymentName:     name,
		EnvironmentName:    environmentName(params.Environment, s.client.profile),
		ObjectCreationType: creationType,
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Queue '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Retrieved Queue", "queue_id", resp.GetQueueId(), "queue_name", name)
	return &Queue{
		QueueID:         resp.GetQueueId(),
		Name:            name,
		cancelEphemeral: nil,
		client:          s.client,
	}, nil
}

// QueueDeleteParams are options for client.Queues.Delete.
type QueueDeleteParams struct {
	Environment  string
	AllowMissing bool
}

// Delete removes a Queue by name.
//
// Warning: Deletion is irreversible and will affect any Apps currently using the Queue.
func (s *queueServiceImpl) Delete(ctx context.Context, name string, params *QueueDeleteParams) error {
	if params == nil {
		params = &QueueDeleteParams{}
	}

	q, err := s.FromName(ctx, name, &QueueFromNameParams{
		Environment:     params.Environment,
		CreateIfMissing: false,
	})

	if err != nil {
		if _, ok := err.(NotFoundError); ok && params.AllowMissing {
			return nil
		}
		return err
	}

	_, err = s.client.cpClient.QueueDelete(ctx, pb.QueueDeleteRequest_builder{QueueId: q.QueueID}.Build())
	if err != nil {
		if st, ok := status.FromError(err); ok && st.Code() == codes.NotFound && params.AllowMissing {
			return nil
		}
		return err
	}

	s.client.logger.DebugContext(ctx, "Deleted Queue", "queue_name", name, "queue_id", q.QueueID)
	return nil
}

type QueueClearParams struct {
	Partition string // partition to clear (default "")
	All       bool   // clear *all* partitions (mutually exclusive with Partition)
}

// Clear removes all objects from a Queue partition.
func (q *Queue) Clear(ctx context.Context, params *QueueClearParams) error {
	if params == nil {
		params = &QueueClearParams{}
	}
	if params.Partition != "" && params.All {
		return InvalidError{"Partition must be \"\" when clearing all partitions"}
	}
	key, err := validatePartitionKey(params.Partition)
	if err != nil {
		return err
	}
	_, err = q.client.cpClient.QueueClear(ctx, pb.QueueClearRequest_builder{
		QueueId:       q.QueueID,
		PartitionKey:  key,
		AllPartitions: params.All,
	}.Build())
	return err
}

// get is an internal helper for both Get and GetMany.
func (q *Queue) get(ctx context.Context, n int, params *QueueGetParams) ([]any, error) {
	if params == nil {
		params = &QueueGetParams{}
	}
	partitionKey, err := validatePartitionKey(params.Partition)
	if err != nil {
		return nil, err
	}

	startTime := time.Now()
	pollTimeout := 50 * time.Second
	if params.Timeout != nil && pollTimeout > *params.Timeout {
		pollTimeout = *params.Timeout
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		resp, err := q.client.cpClient.QueueGet(ctx, pb.QueueGetRequest_builder{
			QueueId:      q.QueueID,
			PartitionKey: partitionKey,
			Timeout:      float32(pollTimeout.Seconds()),
			NValues:      int32(n),
		}.Build())
		if err != nil {
			return nil, err
		}
		if len(resp.GetValues()) > 0 {
			out := make([]any, len(resp.GetValues()))
			for i, raw := range resp.GetValues() {
				v, err := pickleDeserialize(raw)
				if err != nil {
					return nil, err
				}
				out[i] = v
			}
			return out, nil
		}
		if params.Timeout != nil {
			remaining := *params.Timeout - time.Since(startTime)
			if remaining <= 0 {
				message := fmt.Sprintf("Queue %s did not return values within %s", q.QueueID, *params.Timeout)
				return nil, QueueEmptyError{message}
			}
			pollTimeout = min(pollTimeout, remaining)
		}
	}
}

// QueueGetParams are options for Queue.Get.
type QueueGetParams struct {
	Timeout   *time.Duration // wait max (nil = indefinitely)
	Partition string
}

// Get removes and returns one item (blocking by default).
//
// By default, this will wait until at least one item is present in the Queue.
// If `timeout` is set, returns `QueueEmptyError` if no items are available
// within that timeout.
func (q *Queue) Get(ctx context.Context, params *QueueGetParams) (any, error) {
	vals, err := q.get(ctx, 1, params)
	if err != nil {
		return nil, err
	}
	return vals[0], nil // guaranteed len>=1
}

// QueueGetManyParams are options for Queue.GetMany.
type QueueGetManyParams struct {
	QueueGetParams
}

// GetMany removes up to n items.
//
// By default, this will wait until at least one item is present in the Queue.
// If `timeout` is set, returns `QueueEmptyError` if no items are available
// within that timeout.
func (q *Queue) GetMany(ctx context.Context, n int, params *QueueGetManyParams) ([]any, error) {
	if params == nil {
		return q.get(ctx, n, nil)
	}
	return q.get(ctx, n, &params.QueueGetParams)
}

// put is an internal helper for both Put and PutMany.
func (q *Queue) put(ctx context.Context, values []any, params *QueuePutParams) error {
	if params == nil {
		params = &QueuePutParams{}
	}
	key, err := validatePartitionKey(params.Partition)
	if err != nil {
		return err
	}

	valuesEncoded := make([][]byte, len(values))
	for i, v := range values {
		b, err := pickleSerialize(v)
		if err != nil {
			return err
		}
		valuesEncoded[i] = b.Bytes()
	}

	deadline := time.Time{}
	if params.Timeout != nil {
		deadline = time.Now().Add(*params.Timeout)
	}

	delay := queueInitialPutBackoff
	ttl := params.PartitionTTL
	if ttl == 0 {
		ttl = queueDefaultPartitionTTL
	}

	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		_, err := q.client.cpClient.QueuePut(ctx, pb.QueuePutRequest_builder{
			QueueId:             q.QueueID,
			Values:              valuesEncoded,
			PartitionKey:        key,
			PartitionTtlSeconds: int32(ttl.Seconds()),
		}.Build())
		if err == nil {
			return nil // success
		}

		if status.Code(err) != codes.ResourceExhausted {
			return err
		}

		// Queue is full, retry with exponential backoff up to the deadline.
		delay = min(delay*2, 30*time.Second)
		if !deadline.IsZero() {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return QueueFullError{fmt.Sprintf("Put failed on %s", q.QueueID)}
			}
			delay = min(delay, remaining)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}
}

// QueuePutParams are options for Queue.Put.
type QueuePutParams struct {
	Timeout      *time.Duration // max wait for space (nil = indefinitely)
	Partition    string
	PartitionTTL time.Duration // ttl for the *partition* (default 24h)
}

// Put adds a single item to the end of the Queue.
//
// If the Queue is full, this will retry with exponential backoff until the
// provided `timeout` is reached, or indefinitely if `timeout` is not set.
// Raises `QueueFullError` if the Queue is still full after the timeout.
func (q *Queue) Put(ctx context.Context, v any, params *QueuePutParams) error {
	return q.put(ctx, []any{v}, params)
}

// QueuePutManyParams are options for Queue.PutMany.
type QueuePutManyParams struct {
	QueuePutParams
}

// PutMany adds multiple items to the end of the Queue.
//
// If the Queue is full, this will retry with exponential backoff until the
// provided `timeout` is reached, or indefinitely if `timeout` is not set.
// Raises `QueueFullError` if the Queue is still full after the timeout.
func (q *Queue) PutMany(ctx context.Context, values []any, params *QueuePutManyParams) error {
	if params == nil {
		params = &QueuePutManyParams{}
	}
	return q.put(ctx, values, &params.QueuePutParams)
}

type QueueLenParams struct {
	Partition string
	Total     bool // total across all partitions (mutually exclusive with Partition)
}

// Len returns the number of objects in the Queue.
func (q *Queue) Len(ctx context.Context, params *QueueLenParams) (int, error) {
	if params == nil {
		params = &QueueLenParams{}
	}
	if params.Partition != "" && params.Total {
		return 0, InvalidError{"partition must be empty when requesting total length"}
	}
	key, err := validatePartitionKey(params.Partition)
	if err != nil {
		return 0, err
	}
	resp, err := q.client.cpClient.QueueLen(ctx, pb.QueueLenRequest_builder{
		QueueId:      q.QueueID,
		PartitionKey: key,
		Total:        params.Total,
	}.Build())
	if err != nil {
		return 0, err
	}
	return int(resp.GetLen()), nil
}

type QueueIterateParams struct {
	ItemPollTimeout time.Duration // exit if no new items within this period
	Partition       string
}

// Iterate yields items from the Queue until it is empty.
func (q *Queue) Iterate(ctx context.Context, params *QueueIterateParams) iter.Seq2[any, error] {
	if params == nil {
		params = &QueueIterateParams{}
	}

	itemPoll := params.ItemPollTimeout
	lastEntryID := ""
	maxPoll := 30 * time.Second

	return func(yield func(any, error) bool) {
		key, err := validatePartitionKey(params.Partition)
		if err != nil {
			yield(nil, err)
			return
		}

		fetchDeadline := time.Now().Add(itemPoll)
		for {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			pollDuration := max(0, min(maxPoll, time.Until(fetchDeadline)))
			resp, err := q.client.cpClient.QueueNextItems(ctx, pb.QueueNextItemsRequest_builder{
				QueueId:         q.QueueID,
				PartitionKey:    key,
				ItemPollTimeout: float32(pollDuration.Seconds()),
				LastEntryId:     lastEntryID,
			}.Build())
			if err != nil {
				yield(nil, err)
				return
			}
			if len(resp.GetItems()) > 0 {
				for _, item := range resp.GetItems() {
					v, err := pickleDeserialize(item.GetValue())
					if err != nil {
						yield(nil, err)
						return
					}
					if !yield(v, nil) {
						return
					}
					lastEntryID = item.GetEntryId()
				}
				fetchDeadline = time.Now().Add(itemPoll)
			} else if time.Now().After(fetchDeadline) {
				return // exit on idle
			}
		}
	}
}
