package modal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"
	"sync"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/semaphore"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const (
	defaultLogTailEntries          = 100
	maxLogFetchEntries             = 20_000
	logIntervalEntryThreshold      = 2_000
	maxLogFetches                  = 500
	maxLogFetchRange               = 35 * 24 * time.Hour
	maxConcurrentLogFetches        = 10
	maxConcurrentLogCounts         = 20
	maxLogRefinementIterations     = 3
	approximateInitialBuckets      = 100
	logStreamPollInterval          = time.Second
	logStreamRPCTimeout            = 55 * time.Second
	logStreamDrainTimeout          = 500 * time.Millisecond
	logStreamMaxRetries            = 10
	functionCallGetInfoMaxAttempts = 5
)

var logTailLookbacks = [...]time.Duration{
	time.Hour,
	24 * time.Hour,
	7 * 24 * time.Hour,
	30 * 24 * time.Hour,
}

var logBucketSizes = [...]time.Duration{
	2 * time.Second,
	4 * time.Second,
	6 * time.Second,
	12 * time.Second,
	20 * time.Second,
	30 * time.Second,
	time.Minute,
	2 * time.Minute,
	3 * time.Minute,
	4 * time.Minute,
	5 * time.Minute,
	6 * time.Minute,
	10 * time.Minute,
	12 * time.Minute,
	15 * time.Minute,
	20 * time.Minute,
	30 * time.Minute,
	time.Hour,
	2 * time.Hour,
	3 * time.Hour,
	4 * time.Hour,
	8 * time.Hour,
	12 * time.Hour,
	24 * time.Hour,
}

type LogSource string

const (
	LogSourceStdout LogSource = "stdout"
	LogSourceStderr LogSource = "stderr"
	LogSourceSystem LogSource = "system"
)

type logsFilters struct {
	Source         pb.FileDescriptor
	FunctionID     string
	FunctionCallID string
	TaskID         string
	SandboxID      string
	SearchText     string
}

type LogEntry struct {
	Timestamp  time.Time
	Source     LogSource
	Message    string
	ObjectID   string
	ContextIDs []string
}

type FunctionLogsManager struct {
	client     *Client
	appID      string
	functionID string
}

type FunctionCallLogsManager struct {
	client         *Client
	appID          string
	functionID     string
	functionCallID string
}

// FunctionLogFetchParams are options for fetching Function logs.
type FunctionLogFetchParams struct {
	// Until is the end of the time range. It defaults to the current time.
	Until *time.Time
	// Source filters logs by stdout, stderr, or system. The zero value includes all sources.
	Source LogSource
	// SearchText filters Function logs by search text.
	SearchText string
}

// FunctionCallLogFetchParams are options for fetching FunctionCall logs.
type FunctionCallLogFetchParams struct {
	// Since is the start of the time range. It defaults to the start of the FunctionCall.
	Since *time.Time
	// Until is the end of the time range. It defaults to the current time.
	Until *time.Time
	// Source filters logs by stdout, stderr, or system. The zero value includes all sources.
	Source LogSource
	// SearchText filters FunctionCall logs by search text.
	SearchText string
}

// LogTailParams are options for fetching the most recent logs.
type LogTailParams struct {
	// Entries is the number of log entries to return. It defaults to 100.
	Entries int
	// Source filters logs by stdout, stderr, or system. The zero value includes all sources.
	Source LogSource
}

// LogStreamParams are options for streaming logs.
type LogStreamParams struct {
	// Timeout is the duration to wait between log entries before terminating the stream.
	// When nil, the stream blocks until it is interrupted.
	Timeout *time.Duration
}

type resolvedLogFetchParams struct {
	until      time.Time
	source     pb.FileDescriptor
	searchText string
}

func resolveLogFetchParams(
	until *time.Time,
	source LogSource,
	searchText string,
) (resolvedLogFetchParams, error) {
	resolvedUntil := time.Now()
	if until != nil {
		resolvedUntil = *until
	}

	fileDescriptor, err := logSourceFileDescriptor(source)
	if err != nil {
		return resolvedLogFetchParams{}, err
	}
	return resolvedLogFetchParams{
		until:      resolvedUntil,
		source:     fileDescriptor,
		searchText: searchText,
	}, nil
}

func validateLogFetchRange(since time.Time, until time.Time) error {
	if until.Sub(since) > maxLogFetchRange {
		return InvalidError{Exception: fmt.Sprintf(
			"time range must not exceed %d days",
			int(maxLogFetchRange/(24*time.Hour)),
		)}
	}
	return nil
}

// Fetch fetches Function logs corresponding to the date range and filters.
//
// since is the start of the time range. params.Until defaults to the current
// time. The sequence yields [LogEntry] values in chronological order.
func (flm *FunctionLogsManager) Fetch(
	ctx context.Context,
	since time.Time,
	params *FunctionLogFetchParams,
) (iter.Seq2[LogEntry, error], error) {
	var until *time.Time
	var source LogSource
	var searchText string
	if params != nil {
		until = params.Until
		source = params.Source
		searchText = params.SearchText
	}

	resolved, err := resolveLogFetchParams(until, source, searchText)
	if err != nil {
		return nil, err
	}
	if err := validateLogFetchRange(since, resolved.until); err != nil {
		return nil, err
	}

	return fetchLogEntries(
		ctx,
		flm.client,
		flm.appID,
		flm.functionID,
		since,
		resolved.until,
		logsFilters{
			Source:     resolved.source,
			FunctionID: flm.functionID,
			SearchText: resolved.searchText,
		},
	), nil
}

func resolveLogTailParams(params *LogTailParams) (int, pb.FileDescriptor, error) {
	entries := defaultLogTailEntries
	var source LogSource
	if params != nil {
		if params.Entries != 0 {
			entries = params.Entries
		}
		source = params.Source
	}

	if entries < 0 {
		return 0, 0, InvalidError{Exception: "entries must not be negative"}
	}
	if entries > maxLogFetchEntries {
		return 0, 0, InvalidError{Exception: fmt.Sprintf("entries must not exceed %d", maxLogFetchEntries)}
	}

	fileDescriptor, err := logSourceFileDescriptor(source)
	if err != nil {
		return 0, 0, err
	}
	return entries, fileDescriptor, nil
}

// Tail fetches the most recent Function logs.
//
// The sequence yields [LogEntry] values in chronological order.
func (flm *FunctionLogsManager) Tail(ctx context.Context, params *LogTailParams) (iter.Seq2[LogEntry, error], error) {
	entries, source, err := resolveLogTailParams(params)
	if err != nil {
		return nil, err
	}

	return tailLogEntries(
		ctx,
		flm.client,
		flm.appID,
		flm.functionID,
		entries,
		logsFilters{Source: source, FunctionID: flm.functionID},
	), nil
}

func resolveLogStreamParams(params *LogStreamParams) *time.Duration {
	if params == nil || params.Timeout == nil {
		return nil
	}
	timeout := *params.Timeout
	return &timeout
}

// Stream streams new Function logs until the timeout is reached.
//
// The timeout specifies how long to wait between log entries before
// terminating the stream. By default, the stream blocks until it is
// interrupted. The sequence yields [LogEntry] values as they arrive.
func (flm *FunctionLogsManager) Stream(
	ctx context.Context,
	params *LogStreamParams,
) (iter.Seq2[LogEntry, error], error) {
	idleTimeout := resolveLogStreamParams(params)
	return streamLogEntries(
		ctx,
		flm.client,
		flm.appID,
		flm.functionID,
		logsFilters{FunctionID: flm.functionID},
		idleTimeout,
		nil,
	), nil
}

// Fetch fetches all logs associated with the FunctionCall corresponding to the
// date range and filters.
//
// params.Since defaults to the start of the FunctionCall, and params.Until
// defaults to the current time. The sequence yields [LogEntry] values in
// chronological order.
func (fclm *FunctionCallLogsManager) Fetch(
	ctx context.Context,
	params *FunctionCallLogFetchParams,
) (iter.Seq2[LogEntry, error], error) {
	var since time.Time
	var sinceWasProvided bool
	var until *time.Time
	var source LogSource
	var searchText string
	if params != nil {
		if params.Since != nil {
			since = *params.Since
			sinceWasProvided = true
		}
		until = params.Until
		source = params.Source
		searchText = params.SearchText
	}
	untilWasProvided := until != nil

	resolved, err := resolveLogFetchParams(until, source, searchText)
	if err != nil {
		return nil, err
	}
	if !sinceWasProvided {
		info, err := fclm.getFunctionCallInfo(ctx)
		if err != nil {
			return nil, err
		}
		since = unixTime(info.GetCreatedAt())
	}
	if !sinceWasProvided && !untilWasProvided && since.After(resolved.until) {
		resolved.until = since
	}
	if err := validateLogFetchRange(since, resolved.until); err != nil {
		return nil, err
	}

	return fetchLogEntries(
		ctx,
		fclm.client,
		fclm.appID,
		fclm.functionCallID,
		since,
		resolved.until,
		logsFilters{
			Source:         resolved.source,
			FunctionID:     fclm.functionID,
			FunctionCallID: fclm.functionCallID,
			SearchText:     resolved.searchText,
		},
	), nil
}

// Tail fetches the most recent FunctionCall logs.
//
// The sequence yields [LogEntry] values in chronological order.
func (fclm *FunctionCallLogsManager) Tail(ctx context.Context, params *LogTailParams) (iter.Seq2[LogEntry, error], error) {
	entries, source, err := resolveLogTailParams(params)
	if err != nil {
		return nil, err
	}

	return tailLogEntries(
		ctx,
		fclm.client,
		fclm.appID,
		fclm.functionCallID,
		entries,
		logsFilters{
			Source:         source,
			FunctionID:     fclm.functionID,
			FunctionCallID: fclm.functionCallID,
		},
	), nil
}

// Stream streams new FunctionCall logs until the timeout is reached.
//
// The timeout specifies how long to wait between log entries before
// terminating the stream. Stream stops when the FunctionCall is observed to
// have completed or when the timeout is reached. The completion check is best
// effort; if completion cannot be determined, the stream continues until the
// timeout is reached. By default, the stream blocks until it is interrupted.
// The sequence yields [LogEntry] values as they arrive.
func (fclm *FunctionCallLogsManager) Stream(
	ctx context.Context,
	params *LogStreamParams,
) (iter.Seq2[LogEntry, error], error) {
	idleTimeout := resolveLogStreamParams(params)
	return streamLogEntries(
		ctx,
		fclm.client,
		fclm.appID,
		fclm.functionCallID,
		logsFilters{
			FunctionID:     fclm.functionID,
			FunctionCallID: fclm.functionCallID,
		},
		idleTimeout,
		fclm.functionCallComplete,
	), nil
}

type logRange struct {
	start time.Time
	end   time.Time
	count uint64
}

type logInterval struct {
	since time.Time
	until time.Time
}

type logFetchResult struct {
	batches []*pb.TaskLogsBatch
	err     error
}

type logRefinement struct {
	index      int
	bucketSize time.Duration
}

type logRefinementResult struct {
	ranges []logRange
	err    error
}

type logStreamReceive struct {
	batch *pb.TaskLogsBatch
	err   error
}

type logStreamStopFunc func(context.Context) (bool, error)

// logStreamAction is what a caller should do after a log stream attempt fails.
type logStreamAction int

const (
	// logStreamActionFail means the failure is terminal and should be reported.
	// It is the zero value, so an unset action fails rather than looping.
	logStreamActionFail logStreamAction = iota
	// logStreamActionReconnect means the failure is retryable and the caller
	// should open a new attempt.
	logStreamActionReconnect
	// logStreamActionStop means the stopper fired while waiting, so the stream
	// should end through the stop path.
	logStreamActionStop
	// logStreamActionIdle means the idle timeout fired while waiting, so the
	// stream should end quietly.
	logStreamActionIdle
)

// logStreamRetrier bounds how many times a log stream reconnects after a
// retryable failure, and how long it backs off between attempts.
type logStreamRetrier struct {
	remaining int
	delay     time.Duration
}

func newLogStreamRetrier() *logStreamRetrier {
	retrier := &logStreamRetrier{}
	retrier.reset()
	return retrier
}

// reset restores the full retry budget, e.g. once a batch arrives and proves the
// connection healthy again.
func (r *logStreamRetrier) reset() {
	r.remaining = logStreamMaxRetries
	r.delay = time.Millisecond
}

// wait blocks for the current backoff delay and reports what the caller should do
// next. It consumes one retry from the budget when the failure is retryable.
func (r *logStreamRetrier) wait(
	ctx context.Context,
	client *Client,
	streamErr error,
	stopper *logStreamStopper,
	idle *idleLogTimer,
) logStreamAction {
	if ctx.Err() != nil || !isRetryableGrpc(streamErr) || r.remaining <= 0 {
		return logStreamActionFail
	}

	if client.logger != nil {
		client.logger.DebugContext(ctx, "Log stream interrupted; retrying", "error", streamErr)
	}
	r.remaining--

	timer := time.NewTimer(r.delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		r.delay = min(time.Second, r.delay*10)
		return logStreamActionReconnect
	case <-stopper.Done():
		return logStreamActionStop
	case <-idle.C():
		return logStreamActionIdle
	case <-ctx.Done():
		return logStreamActionFail
	}
}

// logStreamStopper watches for the condition that ends a log stream, e.g. the
// function call reaching a terminal state. Done is closed once the stream should
// stop, after which Err reports the watch error, if any. A nil *logStreamStopper
// never stops: Done returns a nil channel, which blocks forever in a select.
type logStreamStopper struct {
	done chan struct{}
	err  error // set before done is closed; read only after observing the close
}

// newLogStreamStopper starts watching stopStream in the background. The returned
// function cancels the watch and waits for it to finish.
func newLogStreamStopper(ctx context.Context, stopStream logStreamStopFunc) (*logStreamStopper, func()) {
	if stopStream == nil {
		return nil, func() {}
	}

	watchCtx, cancelWatch := context.WithCancel(ctx)
	stopper := &logStreamStopper{done: make(chan struct{})}
	var watcher sync.WaitGroup
	watcher.Go(func() {
		stopper.watch(watchCtx, stopStream)
	})

	return stopper, func() {
		cancelWatch()
		watcher.Wait()
	}
}

func (s *logStreamStopper) watch(ctx context.Context, stopStream logStreamStopFunc) {
	for {
		stop, err := stopStream(ctx)
		if err != nil {
			s.err = err
			close(s.done)
			return
		}
		if stop {
			close(s.done)
			return
		}
		if sleepCtx(ctx, logStreamPollInterval) != nil {
			// Canceled: the consumer observes ctx.Done() instead, so leave done
			// open rather than reporting a stop that never happened.
			return
		}
	}
}

func (s *logStreamStopper) Done() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.done
}

func (s *logStreamStopper) Stopped() bool {
	select {
	case <-s.Done():
		return true
	default:
		return false
	}
}

func (s *logStreamStopper) Err() error {
	if s == nil {
		return nil
	}
	return s.err
}

// idleLogTimer fires when no log entries have arrived within its timeout. A nil
// *idleLogTimer, or one built without a timeout, never fires: C returns a nil
// channel, which blocks forever in a select.
type idleLogTimer struct {
	timeout time.Duration
	timer   *time.Timer
}

func newIdleLogTimer(timeout *time.Duration) *idleLogTimer {
	if timeout == nil {
		return &idleLogTimer{}
	}
	return &idleLogTimer{timeout: *timeout, timer: time.NewTimer(*timeout)}
}

func (t *idleLogTimer) C() <-chan time.Time {
	if t == nil || t.timer == nil {
		return nil
	}
	return t.timer.C
}

func (t *idleLogTimer) Reset() {
	if t == nil || t.timer == nil {
		return
	}
	t.timer.Reset(t.timeout)
}

func (t *idleLogTimer) Stop() {
	if t == nil || t.timer == nil {
		return
	}
	t.timer.Stop()
}

func streamLogEntries(
	ctx context.Context,
	client *Client,
	appID string,
	objectID string,
	filters logsFilters,
	idleTimeout *time.Duration,
	stopStream logStreamStopFunc,
) iter.Seq2[LogEntry, error] {
	return func(yield func(LogEntry, error) bool) {
		if idleTimeout != nil && *idleTimeout <= 0 {
			return
		}

		idle := newIdleLogTimer(idleTimeout)
		defer idle.Stop()

		stopper, stopWatching := newLogStreamStopper(ctx, stopStream)
		defer stopWatching()

		lastEntryID := ""
		retrier := newLogStreamRetrier()

		// yieldStop ends the stream the way the stopper asks: reporting its watch
		// error, or draining whatever the stream produced before it stopped. It
		// reads lastEntryID at call time, so it always resumes from the latest
		// entry the caller has seen.
		yieldStop := func() {
			if stopErr := stopper.Err(); stopErr != nil {
				yield(LogEntry{}, stopErr)
				return
			}
			yieldLogStreamDrain(ctx, client, appID, objectID, lastEntryID, filters, yield)
		}

		// retryStream waits out a failed attempt and reports whether to reconnect.
		// Every other outcome ends the iteration, so retryStream yields the final
		// error, if there is one, before returning false.
		retryStream := func(streamErr error) bool {
			switch retrier.wait(ctx, client, streamErr, stopper, idle) {
			case logStreamActionReconnect:
				return true
			case logStreamActionStop:
				yieldStop()
			case logStreamActionIdle:
				// A quiet stream is a normal end of iteration, not an error.
			default:
				if ctx.Err() != nil {
					yield(LogEntry{}, ctx.Err())
				} else {
					yield(LogEntry{}, fmt.Errorf("AppGetLogs failed: %w", streamErr))
				}
			}
			return false
		}

		// Each iteration opens one AppGetLogs attempt; the inner loop consumes it
		// until the attempt ends, then continues here to reconnect.
	attempts:
		for {
			if stopper.Stopped() {
				yieldStop()
				return
			}
			if err := ctx.Err(); err != nil {
				yield(LogEntry{}, err)
				return
			}

			receives, closeAttempt, err := startLogStreamAttempt(
				ctx,
				client,
				buildLogStreamRequest(appID, lastEntryID, logStreamRPCTimeout, filters),
			)
			if err != nil {
				if !retryStream(err) {
					return
				}
				continue
			}

			for {
				select {
				case receive := <-receives:
					if receive.err != nil {
						closeAttempt()
						if errors.Is(receive.err, io.EOF) {
							continue attempts
						}
						if ctx.Err() != nil {
							yield(LogEntry{}, ctx.Err())
							return
						}
						if !retryStream(receive.err) {
							return
						}
						continue attempts
					}

					retrier.reset()
					var keepGoing bool
					lastEntryID, keepGoing = yieldLogStreamBatch(
						receive.batch,
						objectID,
						lastEntryID,
						idle,
						yield,
					)
					if !keepGoing || receive.batch.GetAppDone() {
						closeAttempt()
						return
					}

					if stopper.Stopped() {
						closeAttempt()
						yieldStop()
						return
					}

				case <-idle.C():
					closeAttempt()
					return

				case <-stopper.Done():
					if receive, ready := pollLogStreamReceive(receives); ready {
						if receive.err == nil {
							var keepGoing bool
							lastEntryID, keepGoing = yieldLogStreamBatch(
								receive.batch,
								objectID,
								lastEntryID,
								idle,
								yield,
							)
							if !keepGoing || receive.batch.GetAppDone() {
								closeAttempt()
								return
							}
						} else if !errors.Is(receive.err, io.EOF) && !isRetryableGrpc(receive.err) {
							closeAttempt()
							yield(LogEntry{}, fmt.Errorf("AppGetLogs failed: %w", receive.err))
							return
						}
					}
					closeAttempt()
					yieldStop()
					return

				case <-ctx.Done():
					closeAttempt()
					yield(LogEntry{}, ctx.Err())
					return
				}
			}
		}
	}
}

func startLogStreamAttempt(
	ctx context.Context,
	client *Client,
	request *pb.AppGetLogsRequest,
) (<-chan logStreamReceive, func(), error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	stream, err := client.cpClient.AppGetLogs(attemptCtx, request)
	if err != nil {
		cancel()
		return nil, nil, err
	}

	receives := make(chan logStreamReceive, 1)
	var receiver sync.WaitGroup
	receiver.Go(func() {
		for {
			batch, err := stream.Recv()
			select {
			case receives <- logStreamReceive{batch: batch, err: err}:
			case <-attemptCtx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	})

	closeAttempt := sync.OnceFunc(func() {
		cancel()
		receiver.Wait()
	})
	return receives, closeAttempt, nil
}

func buildLogStreamRequest(
	appID string,
	lastEntryID string,
	timeout time.Duration,
	filters logsFilters,
) *pb.AppGetLogsRequest {
	return pb.AppGetLogsRequest_builder{
		AppId:          appID,
		Timeout:        float32(timeout.Seconds()),
		LastEntryId:    lastEntryID,
		FunctionId:     filters.FunctionID,
		FunctionCallId: filters.FunctionCallID,
		TaskId:         filters.TaskID,
		FileDescriptor: filters.Source,
		SandboxId:      filters.SandboxID,
		SearchText:     filters.SearchText,
	}.Build()
}

func pollLogStreamReceive(receives <-chan logStreamReceive) (logStreamReceive, bool) {
	select {
	case receive := <-receives:
		return receive, true
	default:
		return logStreamReceive{}, false
	}
}

func yieldLogStreamBatch(
	batch *pb.TaskLogsBatch,
	objectID string,
	lastEntryID string,
	idle *idleLogTimer,
	yield func(LogEntry, error) bool,
) (string, bool) {
	if batch.GetEntryId() != "" {
		lastEntryID = batch.GetEntryId()
	}
	for _, item := range batch.GetItems() {
		if item.GetData() == "" {
			continue
		}
		idle.Reset()
		if !yield(logEntryFromItem(item, batch, objectID), nil) {
			return lastEntryID, false
		}
	}
	return lastEntryID, true
}

func yieldLogStreamDrain(
	ctx context.Context,
	client *Client,
	appID string,
	objectID string,
	lastEntryID string,
	filters logsFilters,
	yield func(LogEntry, error) bool,
) {
	drainCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := client.cpClient.AppGetLogs(
		drainCtx,
		buildLogStreamRequest(appID, lastEntryID, logStreamDrainTimeout, filters),
	)
	if err != nil {
		yield(LogEntry{}, fmt.Errorf("AppGetLogs drain failed: %w", err))
		return
	}

	for {
		batch, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return
		}
		if err != nil {
			if ctx.Err() != nil {
				yield(LogEntry{}, ctx.Err())
			} else {
				yield(LogEntry{}, fmt.Errorf("AppGetLogs drain failed: %w", err))
			}
			return
		}

		var keepGoing bool
		lastEntryID, keepGoing = yieldLogStreamBatch(batch, objectID, lastEntryID, nil, yield)
		if !keepGoing || batch.GetAppDone() {
			return
		}
	}
}

func fetchLogEntries(
	ctx context.Context,
	client *Client,
	appID string,
	objectID string,
	since time.Time,
	until time.Time,
	filters logsFilters,
) iter.Seq2[LogEntry, error] {
	return func(yield func(LogEntry, error) bool) {
		for batch, err := range fetchLogBatches(ctx, client, appID, since, until, filters) {
			if err != nil {
				yield(LogEntry{}, err)
				return
			}
			for _, item := range batch.GetItems() {
				if !yield(logEntryFromItem(item, batch, objectID), nil) {
					return
				}
			}
		}
	}
}

func fetchLogBatches(
	ctx context.Context,
	client *Client,
	appID string,
	since time.Time,
	until time.Time,
	filters logsFilters,
) iter.Seq2[*pb.TaskLogsBatch, error] {
	return func(yield func(*pb.TaskLogsBatch, error) bool) {
		bucketSize := pickLogBucketSize(since, until)
		countResponse, err := client.cpClient.AppCountLogs(ctx, buildLogCountRequest(
			appID,
			since,
			until,
			bucketSize,
			filters,
		))
		if err != nil {
			yield(nil, fmt.Errorf("AppCountLogs failed: %w", err))
			return
		}

		ranges := logBucketsToRanges(countResponse.GetBuckets(), bucketSize)
		totalLogs := uint64(0)
		for _, candidateRange := range ranges {
			totalLogs += candidateRange.count
		}
		if totalLogs == 0 {
			return
		}

		for len(ranges) > 0 && ranges[0].count == 0 {
			ranges = ranges[1:]
		}
		for len(ranges) > 0 && ranges[len(ranges)-1].count == 0 {
			ranges = ranges[:len(ranges)-1]
		}

		ranges, err = refineDenseLogRanges(
			ctx,
			client,
			appID,
			ranges,
			filters,
			maxLogFetches,
			maxLogRefinementIterations,
		)
		if err != nil {
			yield(nil, err)
			return
		}

		fetchError := InvalidError{
			Exception: "too many logs to fetch in time range; narrow the range or add filters",
		}
		for _, logRange := range ranges {
			if logRange.count > maxLogFetchEntries {
				yield(nil, fetchError)
				return
			}
		}

		intervals := buildLogFetchIntervals(ranges)
		clampedIntervals := make([]logInterval, 0, len(intervals))
		for _, interval := range intervals {
			interval.since = maxTime(interval.since, since)
			interval.until = minTime(interval.until, until)
			if interval.since.Before(interval.until) {
				clampedIntervals = append(clampedIntervals, interval)
			}
		}
		if len(clampedIntervals) == 0 {
			return
		}
		if len(clampedIntervals) > maxLogFetches {
			yield(nil, fetchError)
			return
		}

		for batch, err := range fetchLogIntervals(ctx, client, appID, clampedIntervals, filters) {
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func pickLogBucketSize(since time.Time, until time.Time) time.Duration {
	duration := until.Sub(since)
	for _, bucketSize := range logBucketSizes {
		if duration <= approximateInitialBuckets*bucketSize {
			return bucketSize
		}
	}
	return logBucketSizes[len(logBucketSizes)-1]
}

func logBucketsToRanges(
	buckets []*pb.AppCountLogsResponse_LogBucket,
	bucketSize time.Duration,
) []logRange {
	ranges := make([]logRange, 0, len(buckets))
	for _, bucket := range buckets {
		start := bucket.GetBucketStartAt().AsTime()
		ranges = append(ranges, logRange{
			start: start,
			end:   start.Add(bucketSize),
			count: bucket.GetStdoutLogs() + bucket.GetStderrLogs() + bucket.GetSystemLogs(),
		})
	}
	return ranges
}

func buildLogFetchIntervals(ranges []logRange) []logInterval {
	intervals := make([]logInterval, 0)
	var currentStart time.Time
	var currentEnd time.Time
	currentCount := uint64(0)

	for _, candidateRange := range ranges {
		if candidateRange.count == 0 {
			if !currentStart.IsZero() {
				intervals = append(intervals, logInterval{since: currentStart, until: currentEnd})
				currentStart = time.Time{}
				currentCount = 0
			}
			continue
		}

		if !currentStart.IsZero() && currentCount+candidateRange.count > maxLogFetchEntries {
			intervals = append(intervals, logInterval{since: currentStart, until: currentEnd})
			currentStart = time.Time{}
			currentCount = 0
		}

		if currentStart.IsZero() {
			currentStart = candidateRange.start
			currentCount = candidateRange.count
		} else {
			currentCount += candidateRange.count
		}
		currentEnd = candidateRange.end

		if currentCount >= logIntervalEntryThreshold {
			intervals = append(intervals, logInterval{since: currentStart, until: currentEnd})
			currentStart = time.Time{}
			currentCount = 0
		}
	}

	if !currentStart.IsZero() {
		intervals = append(intervals, logInterval{since: currentStart, until: currentEnd})
	}
	return intervals
}

func nextSmallerLogBucketSize(current time.Duration) (time.Duration, bool) {
	for index := len(logBucketSizes) - 1; index >= 0; index-- {
		if logBucketSizes[index] < current {
			return logBucketSizes[index], true
		}
	}
	return 0, false
}

func refineDenseLogRanges(
	ctx context.Context,
	client *Client,
	appID string,
	ranges []logRange,
	filters logsFilters,
	maxRanges int,
	maxIterations int,
) ([]logRange, error) {
	refined := append([]logRange(nil), ranges...)

	for range maxIterations {
		var refinements []logRefinement
		for index, candidateRange := range refined {
			if candidateRange.count <= maxLogFetchEntries {
				continue
			}
			bucketSize, ok := nextSmallerLogBucketSize(candidateRange.end.Sub(candidateRange.start))
			if ok {
				refinements = append(refinements, logRefinement{
					index:      index,
					bucketSize: bucketSize,
				})
			}
		}
		if len(refinements) == 0 {
			break
		}

		estimatedRangeCount := len(refined)
		for _, refinement := range refinements {
			duration := refined[refinement.index].end.Sub(refined[refinement.index].start)
			subrangeCount := int((duration + refinement.bucketSize - 1) / refinement.bucketSize)
			estimatedRangeCount += subrangeCount - 1
		}
		if estimatedRangeCount > maxRanges {
			break
		}

		results := runLogRefinements(ctx, client, appID, refined, refinements, filters)
		refinementByIndex := make(map[int]int, len(refinements))
		for resultIndex, refinement := range refinements {
			refinementByIndex[refinement.index] = resultIndex
		}

		nextRefined := make([]logRange, 0, estimatedRangeCount)
		for index, candidateRange := range refined {
			resultIndex, ok := refinementByIndex[index]
			if !ok {
				nextRefined = append(nextRefined, candidateRange)
				continue
			}
			result := results[resultIndex]
			if result.err != nil {
				return nil, result.err
			}
			nextRefined = append(nextRefined, result.ranges...)
		}
		refined = nextRefined
	}

	return refined, nil
}

func runLogRefinements(
	ctx context.Context,
	client *Client,
	appID string,
	ranges []logRange,
	refinements []logRefinement,
	filters logsFilters,
) []logRefinementResult {
	results := make([]logRefinementResult, len(refinements))
	sem := semaphore.NewWeighted(maxConcurrentLogCounts)
	var workers sync.WaitGroup

	for resultIndex, refinement := range refinements {
		parent := ranges[refinement.index]
		workers.Add(1)

		go func() {
			defer workers.Done()
			if err := sem.Acquire(ctx, 1); err != nil {
				results[resultIndex] = logRefinementResult{err: err}
				return
			}
			defer sem.Release(1)

			response, err := client.cpClient.AppCountLogs(ctx, buildLogCountRequest(
				appID,
				parent.start,
				parent.end,
				refinement.bucketSize,
				filters,
			))
			if err != nil {
				results[resultIndex] = logRefinementResult{err: fmt.Errorf("AppCountLogs failed: %w", err)}
				return
			}

			subranges := logBucketsToRanges(response.GetBuckets(), refinement.bucketSize)
			if len(subranges) > 0 {
				first := &subranges[0]
				first.start = maxTime(first.start, parent.start)
				last := &subranges[len(subranges)-1]
				last.end = minTime(last.end, parent.end)
			}

			results[resultIndex] = logRefinementResult{ranges: subranges}
		}()
	}
	workers.Wait()
	return results
}

func buildLogCountRequest(
	appID string,
	since time.Time,
	until time.Time,
	bucketSize time.Duration,
	filters logsFilters,
) *pb.AppCountLogsRequest {
	return pb.AppCountLogsRequest_builder{
		AppId:          appID,
		TaskId:         filters.TaskID,
		FunctionId:     filters.FunctionID,
		FunctionCallId: filters.FunctionCallID,
		SandboxId:      filters.SandboxID,
		SearchText:     filters.SearchText,
		Since:          timestamppb.New(since),
		Until:          timestamppb.New(until),
		BucketSecs:     uint32(bucketSize / time.Second),
		Source:         filters.Source,
	}.Build()
}

func fetchLogIntervals(
	ctx context.Context,
	client *Client,
	appID string,
	intervals []logInterval,
	filters logsFilters,
) iter.Seq2[*pb.TaskLogsBatch, error] {
	return func(yield func(*pb.TaskLogsBatch, error) bool) {
		fetchCtx, cancel := context.WithCancel(ctx)
		sem := semaphore.NewWeighted(maxConcurrentLogFetches)
		results := make([]chan logFetchResult, len(intervals))
		for index := range results {
			results[index] = make(chan logFetchResult, 1)
		}

		var workers sync.WaitGroup
		schedulerDone := make(chan struct{})

		defer func() {
			cancel()
			<-schedulerDone
			workers.Wait()
		}()

		go func() {
			defer close(schedulerDone)
			for index, interval := range intervals {
				if err := sem.Acquire(fetchCtx, 1); err != nil {
					results[index] <- logFetchResult{err: err}
					return
				}
				workers.Add(1)

				go func() {
					defer workers.Done()
					defer sem.Release(1)

					response, err := client.cpClient.AppFetchLogs(fetchCtx, pb.AppFetchLogsRequest_builder{
						AppId:          appID,
						Since:          timestamppb.New(interval.since),
						Until:          timestamppb.New(interval.until),
						Limit:          maxLogFetchEntries,
						Source:         filters.Source,
						FunctionId:     filters.FunctionID,
						FunctionCallId: filters.FunctionCallID,
						TaskId:         filters.TaskID,
						SandboxId:      filters.SandboxID,
						SearchText:     filters.SearchText,
					}.Build())
					if err != nil {
						results[index] <- logFetchResult{err: fmt.Errorf("AppFetchLogs failed: %w", err)}
						return
					}
					results[index] <- logFetchResult{batches: response.GetBatches()}
				}()
			}
		}()

		for _, resultChannel := range results {
			var result logFetchResult
			select {
			case result = <-resultChannel:
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			}

			if result.err != nil {
				yield(nil, result.err)
				return
			}
			for _, batch := range result.batches {
				if !yield(batch, nil) {
					return
				}
			}
		}
	}
}

func (fclm *FunctionCallLogsManager) getFunctionCallInfo(ctx context.Context) (*pb.FunctionCallInfo, error) {
	request := pb.FunctionCallGetInfoRequest_builder{
		FunctionId:     fclm.functionID,
		FunctionCallId: fclm.functionCallID,
	}.Build()

	for attempt := range functionCallGetInfoMaxAttempts {
		response, err := fclm.client.cpClient.FunctionCallGetInfo(ctx, request)
		if err == nil {
			return response.GetInfo(), nil
		}
		if status.Code(err) != codes.NotFound {
			return nil, fmt.Errorf("FunctionCallGetInfo failed: %w", err)
		}
		if attempt == functionCallGetInfoMaxAttempts-1 {
			return nil, NotFoundError{Exception: fmt.Sprintf("Function call '%s' not found", fclm.functionCallID)}
		}
		if err := sleepCtx(ctx, time.Second); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("FunctionCallGetInfo failed after %d attempts", functionCallGetInfoMaxAttempts)
}

func (fclm *FunctionCallLogsManager) functionCallComplete(ctx context.Context) (bool, error) {
	info, err := fclm.getFunctionCallInfo(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		var notFoundError NotFoundError
		if errors.As(err, &notFoundError) {
			return false, nil
		}
		switch status.Code(err) {
		case codes.ResourceExhausted:
			if err := sleepCtx(ctx, time.Second); err != nil {
				return false, err
			}
			return false, nil
		case codes.NotFound:
			return false, nil
		default:
			if isRetryableGrpc(err) {
				if fclm.client.logger != nil {
					fclm.client.logger.DebugContext(ctx, "FunctionCall status check interrupted; retrying", "error", err)
				}
				return false, nil
			}
			return false, err
		}
	}

	terminalInputs := info.GetSucceededInputs().GetTotal() +
		info.GetFailedInputs().GetTotal() +
		info.GetTimeoutInputs().GetTotal() +
		info.GetCancelledInputs().GetTotal()
	return info.GetTotalInputs() > 0 && terminalInputs == info.GetTotalInputs(), nil
}

func maxTime(left time.Time, right time.Time) time.Time {
	if left.After(right) {
		return left
	}
	return right
}

func minTime(left time.Time, right time.Time) time.Time {
	if left.Before(right) {
		return left
	}
	return right
}

func tailLogEntries(
	ctx context.Context,
	client *Client,
	appID string,
	objectID string,
	entries int,
	filters logsFilters,
) iter.Seq2[LogEntry, error] {
	return func(yield func(LogEntry, error) bool) {
		for batch, err := range tailLogBatches(ctx, client, appID, entries, filters) {
			if err != nil {
				yield(LogEntry{}, err)
				return
			}
			for _, item := range batch.GetItems() {
				if !yield(logEntryFromItem(item, batch, objectID), nil) {
					return
				}
			}
		}
	}
}

func tailLogBatches(
	ctx context.Context,
	client *Client,
	appID string,
	entries int,
	filters logsFilters,
) iter.Seq2[*pb.TaskLogsBatch, error] {
	return func(yield func(*pb.TaskLogsBatch, error) bool) {
		until := time.Now()

		for _, lookback := range logTailLookbacks {
			request := pb.AppFetchLogsRequest_builder{
				AppId:          appID,
				Since:          timestamppb.New(until.Add(-lookback)),
				Until:          timestamppb.New(until),
				Limit:          uint32(entries),
				Source:         filters.Source,
				FunctionId:     filters.FunctionID,
				FunctionCallId: filters.FunctionCallID,
				TaskId:         filters.TaskID,
				SandboxId:      filters.SandboxID,
				SearchText:     filters.SearchText,
			}.Build()

			response, err := client.cpClient.AppFetchLogs(ctx, request)
			if err != nil {
				yield(nil, fmt.Errorf("AppFetchLogs failed: %w", err))
				return
			}

			totalItems := 0
			for _, batch := range response.GetBatches() {
				totalItems += len(batch.GetItems())
			}
			if totalItems < entries && lookback != logTailLookbacks[len(logTailLookbacks)-1] {
				continue
			}

			for _, batch := range response.GetBatches() {
				if !yield(batch, nil) {
					return
				}
			}
			return
		}
	}
}

func logSourceFileDescriptor(source LogSource) (pb.FileDescriptor, error) {
	switch source {
	case "":
		return pb.FileDescriptor_FILE_DESCRIPTOR_UNSPECIFIED, nil
	case LogSourceStdout:
		return pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT, nil
	case LogSourceStderr:
		return pb.FileDescriptor_FILE_DESCRIPTOR_STDERR, nil
	case LogSourceSystem:
		return pb.FileDescriptor_FILE_DESCRIPTOR_INFO, nil
	default:
		return pb.FileDescriptor_FILE_DESCRIPTOR_UNSPECIFIED, InvalidError{
			Exception: `source must be one of "stdout", "stderr", "system", or empty`,
		}
	}
}

func logEntryFromItem(item *pb.TaskLogs, batch *pb.TaskLogsBatch, objectID string) LogEntry {
	return LogEntry{
		Timestamp:  logEntryTimestamp(item),
		Source:     logEntrySource(item.GetFileDescriptor()),
		Message:    item.GetData(),
		ObjectID:   objectID,
		ContextIDs: logEntryContextIDs(item, batch, objectID),
	}
}

func logEntryTimestamp(item *pb.TaskLogs) time.Time {
	if timestampNS := item.GetTimestampNs(); timestampNS != 0 {
		return time.Unix(0, int64(timestampNS)).UTC()
	}
	return unixTime(item.GetTimestamp())
}

func unixTime(timestamp float64) time.Time {
	seconds := int64(timestamp)
	nanoseconds := int64((timestamp - float64(seconds)) * float64(time.Second))
	return time.Unix(seconds, nanoseconds).UTC()
}

func logEntrySource(fileDescriptor pb.FileDescriptor) LogSource {
	switch fileDescriptor {
	case pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT:
		return LogSourceStdout
	case pb.FileDescriptor_FILE_DESCRIPTOR_STDERR:
		return LogSourceStderr
	default:
		return LogSourceSystem
	}
}

func logEntryContextIDs(item *pb.TaskLogs, batch *pb.TaskLogsBatch, objectID string) []string {
	var contextIDs []string
	switch {
	case strings.HasPrefix(objectID, "fu-"):
		contextIDs = []string{
			item.GetFunctionCallId(),
			firstNonEmpty(item.GetInputId(), batch.GetInputId()),
			firstNonEmpty(item.GetContainerId(), batch.GetTaskId()),
		}
	case strings.HasPrefix(objectID, "fc-"):
		contextIDs = []string{
			firstNonEmpty(item.GetInputId(), batch.GetInputId()),
			firstNonEmpty(item.GetContainerId(), batch.GetTaskId()),
		}
	}

	nonempty := contextIDs[:0]
	for _, contextID := range contextIDs {
		if contextID != "" {
			nonempty = append(nonempty, contextID)
		}
	}
	return nonempty
}
