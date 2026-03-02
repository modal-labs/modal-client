package test

import (
	"context"
	"net/http"
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	"github.com/onsi/gomega"
)

func newTestClient(t *testing.T) *modal.Client {
	t.Helper()

	c, err := modal.NewClient()
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		c.Close()

		// Close idle http connections to silence goleak.
		http.DefaultClient.CloseIdleConnections()
	})

	return c
}

func newGRPCMockClient(t *testing.T) *grpcmock.MockClient {
	t.Helper()

	mock := grpcmock.NewMockClient()

	t.Cleanup(func() {
		mock.Close()
	})

	return mock
}

func terminateSandbox(g *gomega.WithT, sb *modal.Sandbox) {
	_, err := sb.Terminate(context.Background(), nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}
