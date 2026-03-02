package modal

import (
	"context"
	"fmt"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ProxyService provides Proxy related operations.
type ProxyService interface {
	FromName(ctx context.Context, name string, params *ProxyFromNameParams) (*Proxy, error)
}

type proxyServiceImpl struct{ client *Client }

// Proxy represents a Modal Proxy.
type Proxy struct {
	ProxyID string
}

// ProxyFromNameParams are options for looking up a Modal Proxy.
type ProxyFromNameParams struct {
	Environment string
}

// FromName references a modal.Proxy by its name.
func (s *proxyServiceImpl) FromName(ctx context.Context, name string, params *ProxyFromNameParams) (*Proxy, error) {
	if params == nil {
		params = &ProxyFromNameParams{}
	}

	resp, err := s.client.cpClient.ProxyGet(ctx, pb.ProxyGetRequest_builder{
		Name:            name,
		EnvironmentName: environmentName(params.Environment, s.client.profile),
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Proxy '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}

	if resp.GetProxy() == nil || resp.GetProxy().GetProxyId() == "" {
		return nil, NotFoundError{fmt.Sprintf("Proxy '%s' not found", name)}
	}

	return &Proxy{ProxyID: resp.GetProxy().GetProxyId()}, nil
}
