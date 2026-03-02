// Package grpcmock provides utilities for mocking gRPC services in tests.
package grpcmock

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	modal "github.com/modal-labs/modal-client/go"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
)

// unaryHandler handles a single unary RPC request and returns a response.
type unaryHandler func(proto.Message) (proto.Message, error)

// MockClient wraps modal.Client with mock capabilities
type MockClient struct {
	*modal.Client
	// mu guards access to internal state.
	mu sync.Mutex
	// methodHandlerQueues maps short RPC names to FIFO queues of handlers.
	methodHandlerQueues map[string][]unaryHandler
}

// NewMockClient creates a Modal client with mock backends for testing.
func NewMockClient() *MockClient {
	mc := &MockClient{
		methodHandlerQueues: make(map[string][]unaryHandler),
	}

	conn := &mockClientConn{mock: mc}

	modalClient, err := modal.NewClientWithOptions(&modal.ClientParams{
		TokenID:            "test-token-id",
		TokenSecret:        "test-token-secret",
		Environment:        "test",
		ControlPlaneClient: pb.NewModalClientClient(conn),
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create mock client: %v", err))
	}

	mc.Client = modalClient
	return mc
}

// HandleUnary registers a typed handler for a unary RPC, e.g. "/FunctionGetCurrentStats".
func HandleUnary[Req proto.Message, Resp proto.Message](m *MockClient, rpc string, handler func(Req) (Resp, error)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	name := shortName(rpc)
	q := m.methodHandlerQueues[name]
	wrapped := unaryHandler(func(in proto.Message) (proto.Message, error) {
		req, ok := any(in).(Req)
		if !ok {
			return nil, fmt.Errorf("grpcmock: request type mismatch for %s: expected %T, got %T", name, *new(Req), in)
		}
		resp, err := handler(req)
		if err != nil {
			return nil, err
		}
		var out proto.Message = resp
		return out, nil
	})
	m.methodHandlerQueues[name] = append(q, wrapped)
}

// AssertExhausted errors unless all registered mock expectations have been consumed.
func (m *MockClient) AssertExhausted() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	var outstanding []string
	for k, q := range m.methodHandlerQueues {
		if len(q) > 0 {
			outstanding = append(outstanding, fmt.Sprintf("%s: %d remaining", k, len(q)))
		}
	}
	if len(outstanding) > 0 {
		return fmt.Errorf("not all expected gRPC calls were made:\n- %s", strings.Join(outstanding, "\n- "))
	}
	return nil
}

// mockClientConn implements grpc.ClientConnInterface for unary calls.
type mockClientConn struct{ mock *MockClient }

// Invoke implements grpc.ClientConnInterface.Invoke for unary RPCs.
func (c *mockClientConn) Invoke(ctx context.Context, method string, in, out any, opts ...grpc.CallOption) error {
	name := shortName(method)
	handler, err := c.dequeueNextHandler(name)
	if err != nil {
		return err
	}
	resp, err := handler(in.(proto.Message))
	if err != nil {
		return err
	}
	if resp != nil {
		if outMsg, ok := out.(proto.Message); ok {
			proto.Merge(outMsg, resp)
		} else {
			return fmt.Errorf("grpcmock: response cannot be written into type %T", out)
		}
	}
	return nil
}

// NewStream returns an error because streaming RPCs are not supported yet.
func (c *mockClientConn) NewStream(ctx context.Context, desc *grpc.StreamDesc, method string, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	return nil, fmt.Errorf("grpcmock: streaming not implemented for %s", shortName(method))
}

func (c *mockClientConn) dequeueNextHandler(method string) (unaryHandler, error) {
	c.mock.mu.Lock()
	defer c.mock.mu.Unlock()
	q := c.mock.methodHandlerQueues[method]
	if len(q) == 0 {
		return nil, fmt.Errorf("grpcmock: unexpected gRPC call to %s", method)
	}
	h := q[0]
	c.mock.methodHandlerQueues[method] = q[1:]
	return h, nil
}

func shortName(method string) string {
	if strings.HasPrefix(method, "/") {
		if idx := strings.LastIndex(method, "/"); idx >= 0 && idx+1 < len(method) {
			return method[idx+1:]
		}
	}
	return method
}
