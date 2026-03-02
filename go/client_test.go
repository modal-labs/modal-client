package modal

import (
	"context"
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"

	"github.com/onsi/gomega"
	"google.golang.org/grpc"
)

func TestClientWithLogger(t *testing.T) {
	g := gomega.NewWithT(t)

	r, w, err := os.Pipe()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	logger := slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: slog.LevelDebug}))
	g.Expect(logger).NotTo(gomega.BeNil())

	client, err := NewClientWithOptions(&ClientParams{Logger: logger})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(client).NotTo(gomega.BeNil())

	err = w.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(r)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(output).To(gomega.ContainSubstring("Initializing Modal client"))
	g.Expect(output).To(gomega.ContainSubstring("Modal client initialized successfully"))
}

func TestClientWithCustomInterceptors(t *testing.T) {
	g := gomega.NewWithT(t)

	var firstCalled, secondCalled bool
	var firstMethod, secondMethod string
	var mu sync.Mutex

	firstInterceptor := func(
		ctx context.Context,
		method string,
		req, reply any,
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		mu.Lock()
		firstCalled = true
		firstMethod = method
		mu.Unlock()
		return invoker(ctx, method, req, reply, cc, opts...)
	}

	secondInterceptor := func(
		ctx context.Context,
		method string,
		req, reply any,
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		mu.Lock()
		secondCalled = true
		secondMethod = method
		mu.Unlock()
		return invoker(ctx, method, req, reply, cc, opts...)
	}

	client, err := NewClientWithOptions(&ClientParams{
		GRPCUnaryInterceptors: []grpc.UnaryClientInterceptor{firstInterceptor, secondInterceptor},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(client).NotTo(gomega.BeNil())
	defer client.Close()

	_, err = client.Functions.FromName(t.Context(), "libmodal-test-support", "non-existent", nil)
	g.Expect(err).Should(gomega.HaveOccurred()) // don't care about success here, just need the RPC

	mu.Lock()
	g.Expect(firstCalled).To(gomega.BeTrue())
	g.Expect(firstMethod).To(gomega.ContainSubstring("ModalClient/"))
	g.Expect(secondCalled).To(gomega.BeTrue())
	g.Expect(secondMethod).To(gomega.ContainSubstring("ModalClient/"))
	mu.Unlock()
}
