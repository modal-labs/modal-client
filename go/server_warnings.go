package modal

import (
	"context"
	"log/slog"
	"net/url"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// serverWarningHeader is the trailing metadata key the server uses to attach a non-fatal message to
// any response, one percent-encoded entry per warning.
const serverWarningHeader = "x-modal-warning"

const serverWarningRegistryLimit = 2048

// serverWarningRegistry suppresses repeated server warnings, best-effort: it is cleared on overflow,
// so a message may be logged again.
type serverWarningRegistry struct {
	mu   sync.Mutex
	seen map[string]struct{}
}

// shouldLog reports whether message has not been logged yet, and records it.
func (r *serverWarningRegistry) shouldLog(message string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.seen[message]; ok {
		return false
	}
	// Start over instead of going quiet: overflow costs a repeat, never a hidden warning.
	if r.seen == nil || len(r.seen) >= serverWarningRegistryLimit {
		r.seen = make(map[string]struct{})
	}
	r.seen[message] = struct{}{}
	return true
}

// serverWarningLogger logs the warnings a server attaches to its responses. One may be shared by
// several connections so a warning repeated across them is suppressed too.
type serverWarningLogger struct {
	logger   *slog.Logger
	registry serverWarningRegistry
}

func newServerWarningLogger(logger *slog.Logger) *serverWarningLogger {
	return &serverWarningLogger{logger: logger}
}

func (w *serverWarningLogger) log(ctx context.Context, md metadata.MD) {
	for _, encoded := range md.Get(serverWarningHeader) {
		message, err := url.PathUnescape(encoded)
		if err != nil {
			message = encoded
		}
		if w.registry.shouldLog(message) {
			w.logger.WarnContext(ctx, message)
		}
	}
}

// serverWarningUnaryInterceptor logs warnings the server attached to a unary response, including
// failed ones.
func serverWarningUnaryInterceptor(w *serverWarningLogger) grpc.UnaryClientInterceptor {
	return func(
		ctx context.Context,
		method string,
		req, reply any,
		cc *grpc.ClientConn,
		inv grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		var trailer metadata.MD
		err := inv(ctx, method, req, reply, cc, append(opts, grpc.Trailer(&trailer))...)
		w.log(ctx, trailer)
		return err
	}
}

// serverWarningStreamInterceptor logs warnings the server attached to a stream once it finishes.
func serverWarningStreamInterceptor(w *serverWarningLogger) grpc.StreamClientInterceptor {
	return func(
		ctx context.Context,
		desc *grpc.StreamDesc,
		cc *grpc.ClientConn,
		method string,
		streamer grpc.Streamer,
		opts ...grpc.CallOption,
	) (grpc.ClientStream, error) {
		cs, err := streamer(ctx, desc, cc, method, opts...)
		if err != nil {
			return nil, err
		}
		return &serverWarningClientStream{ClientStream: cs, ctx: ctx, warnings: w, serverStreams: desc.ServerStreams}, nil
	}
}

type serverWarningClientStream struct {
	grpc.ClientStream
	ctx           context.Context
	warnings      *serverWarningLogger
	serverStreams bool
	once          sync.Once
}

func (s *serverWarningClientStream) RecvMsg(m any) error {
	err := s.ClientStream.RecvMsg(m)
	// Trailers are only available once the stream is done, which for a single-response stream is
	// after the first RecvMsg.
	if err != nil || !s.serverStreams {
		s.once.Do(func() { s.warnings.log(s.ctx, s.Trailer()) })
	}
	return err
}
