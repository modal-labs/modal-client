package modal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func newServerWarningTestLogger() (*serverWarningLogger, *bytes.Buffer) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn}))
	return newServerWarningLogger(logger), &buf
}

func invokerWithTrailer(md metadata.MD, err error) grpc.UnaryInvoker {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		for _, o := range opts {
			if t, ok := o.(grpc.TrailerCallOption); ok {
				*t.TrailerAddr = md
			}
		}
		return err
	}
}

func TestServerWarningUnaryInterceptor(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	w, buf := newServerWarningTestLogger()
	md := metadata.Pairs(serverWarningHeader, "Hello%20world", serverWarningHeader, "100%25%20done")
	err := serverWarningUnaryInterceptor(w)(t.Context(), "/m", nil, nil, nil, invokerWithTrailer(md, nil))
	g.Expect(err).ToNot(gomega.HaveOccurred())

	out := buf.String()
	g.Expect(out).To(gomega.ContainSubstring(`level=WARN msg="Hello world"`))
	g.Expect(out).To(gomega.ContainSubstring(`level=WARN msg="100% done"`))
}

func TestServerWarningUnaryInterceptorOnError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	w, buf := newServerWarningTestLogger()
	md := metadata.Pairs(serverWarningHeader, "Something%20happened")
	rpcErr := status.Error(codes.Internal, "boom")
	err := serverWarningUnaryInterceptor(w)(t.Context(), "/m", nil, nil, nil, invokerWithTrailer(md, rpcErr))
	g.Expect(err).To(gomega.Equal(rpcErr))
	g.Expect(buf.String()).To(gomega.ContainSubstring(`msg="Something happened"`))
}

func TestServerWarningUnaryInterceptorNoWarnings(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	w, buf := newServerWarningTestLogger()
	md := metadata.Pairs("x-other", "value")
	err := serverWarningUnaryInterceptor(w)(t.Context(), "/m", nil, nil, nil, invokerWithTrailer(md, nil))
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(buf.String()).To(gomega.BeEmpty())
}

type fakeTrailerStream struct {
	grpc.ClientStream
	trailer metadata.MD
}

func (s *fakeTrailerStream) RecvMsg(any) error    { return io.EOF }
func (s *fakeTrailerStream) Trailer() metadata.MD { return s.trailer }

func TestServerWarningStreamInterceptor(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	w, buf := newServerWarningTestLogger()
	fake := &fakeTrailerStream{trailer: metadata.Pairs(serverWarningHeader, "Stream%20warning")}
	streamer := func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		return fake, nil
	}
	cs, err := serverWarningStreamInterceptor(w)(t.Context(), &grpc.StreamDesc{ServerStreams: true}, nil, "/m", streamer)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(errors.Is(cs.RecvMsg(nil), io.EOF)).To(gomega.BeTrue())
	g.Expect(errors.Is(cs.RecvMsg(nil), io.EOF)).To(gomega.BeTrue())
	g.Expect(strings.Count(buf.String(), `msg="Stream warning"`)).To(gomega.Equal(1))
}

func TestServerWarningRepeatsSuppressed(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	w, buf := newServerWarningTestLogger()
	same := metadata.Pairs(serverWarningHeader, "Same%20warning")
	for range 5 {
		err := serverWarningUnaryInterceptor(w)(t.Context(), "/m", nil, nil, nil, invokerWithTrailer(same, nil))
		g.Expect(err).ToNot(gomega.HaveOccurred())
	}
	other := metadata.Pairs(serverWarningHeader, "Other%20warning")
	err := serverWarningUnaryInterceptor(w)(t.Context(), "/m", nil, nil, nil, invokerWithTrailer(other, nil))
	g.Expect(err).ToNot(gomega.HaveOccurred())

	out := buf.String()
	g.Expect(strings.Count(out, `msg="Same warning"`)).To(gomega.Equal(1))
	g.Expect(strings.Count(out, `msg="Other warning"`)).To(gomega.Equal(1))
}

func TestServerWarningRepeatsSuppressedAcrossStreams(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	w, buf := newServerWarningTestLogger()
	streamer := func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		return &fakeTrailerStream{trailer: metadata.Pairs(serverWarningHeader, "Stream%20warning")}, nil
	}
	for range 2 {
		cs, err := serverWarningStreamInterceptor(w)(t.Context(), &grpc.StreamDesc{ServerStreams: true}, nil, "/m", streamer)
		g.Expect(err).ToNot(gomega.HaveOccurred())
		g.Expect(errors.Is(cs.RecvMsg(nil), io.EOF)).To(gomega.BeTrue())
	}
	g.Expect(strings.Count(buf.String(), `msg="Stream warning"`)).To(gomega.Equal(1))
}

func TestServerWarningRegistryBounded(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var r serverWarningRegistry
	logged := 0
	for i := range serverWarningRegistryLimit * 3 {
		if r.shouldLog(fmt.Sprintf("warning about object %d", i)) {
			logged++
		}
	}
	g.Expect(logged).To(gomega.Equal(serverWarningRegistryLimit * 3))
	g.Expect(len(r.seen)).To(gomega.BeNumerically("<=", serverWarningRegistryLimit))
	// Overflow drops the entry for this message rather than silencing it.
	g.Expect(r.shouldLog("warning about object 0")).To(gomega.BeTrue())
}
