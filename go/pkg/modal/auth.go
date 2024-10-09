package modal

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

type AuthedClient struct {
	Client      ModalClientClient
	Interceptor *AuthInterceptor
}

type AuthField struct {
	Key   string
	Value string
}

type AuthInterceptor struct {
	ClientID     *AuthField
	ClientSecret *AuthField
	Workspace    *AuthField
}

func NewAuthInterceptor(clientID, clientSecret, workspace *AuthField) *AuthInterceptor {
	return &AuthInterceptor{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		Workspace:    workspace,
	}
}

func (ai *AuthInterceptor) Unary() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx = ai.addMetadata(ctx)
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

func (ai *AuthInterceptor) Stream() grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx = ai.addMetadata(ctx)
		return streamer(ctx, desc, cc, method, opts...)
	}
}

func (ai *AuthInterceptor) addMetadata(ctx context.Context) context.Context {
	md := metadata.New(map[string]string{})
	if ai.ClientID != nil {
		md.Set(ai.ClientID.Key, ai.ClientID.Value)
	}
	if ai.ClientSecret != nil {
		md.Set(ai.ClientSecret.Key, ai.ClientSecret.Value)
	}
	if ai.Workspace != nil {
		md.Set(ai.Workspace.Key, ai.Workspace.Value)
	}
	return metadata.NewOutgoingContext(ctx, md)
}

func NewAuthedClient(conn *grpc.ClientConn, interceptor *AuthInterceptor) *AuthedClient {
	return &AuthedClient{
		Client:      NewModalClientClient(conn),
		Interceptor: interceptor,
	}
}
