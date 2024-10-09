package modal

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

type Client struct {
	client ModalClientClient
}

func Connect(config Config, workspace string) (*Client, error) {
	var clientID, clientSecret string
	var clientIDHeader, clientSecretHeader string

	if config.TaskID != nil && *config.TaskID != "" && config.TaskSecret != nil && *config.TaskSecret != "" {
		clientID = *config.TaskID
		clientSecret = *config.TaskSecret
		clientIDHeader = "x-modal-task-id"
		clientSecretHeader = "x-modal-task-secret"
	} else if config.TokenID != nil && *config.TokenID != "" && config.TokenSecret != nil && *config.TokenSecret != "" {
		clientID = *config.TokenID
		clientSecret = *config.TokenSecret
		clientIDHeader = "x-modal-token-id"
		clientSecretHeader = "x-modal-token-secret"
	}

	var serverURL string
	if config.ServerURL != nil && *config.ServerURL != "" {
		serverURL = *config.ServerURL
	} else {
		serverURL = "https://api.modal.com"
	}

	// Parse the server URL
	parsedURL, err := url.Parse(serverURL)
	if err != nil {
		return nil, fmt.Errorf("invalid server URL: %v", err)
	}

	// Ensure the scheme is set
	if parsedURL.Scheme == "" {
		parsedURL.Scheme = "https"
	}

	// Remove the scheme for gRPC dial
	host := parsedURL.Host
	if !strings.Contains(host, ":") {
		if parsedURL.Scheme == "https" {
			host += ":443"
		} else {
			host += ":80"
		}
	}

	// Create a context with timeout for the connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Set up the gRPC options
	var opts []grpc.DialOption

	// Determine whether to use TLS based on the URL scheme
	if parsedURL.Scheme == "https" {
		creds := credentials.NewClientTLSFromCert(nil, "")
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	// Add the interceptor
	opts = append(opts, grpc.WithUnaryInterceptor(func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		md := metadata.New(map[string]string{
			clientIDHeader:     clientID,
			clientSecretHeader: clientSecret,
		})
		if workspace != "" {
			md.Set("x-modal-workspace", workspace)
		}
		ctx = metadata.NewOutgoingContext(ctx, md)
		return invoker(ctx, method, req, reply, cc, opts...)
	}))

	// Create the connection
	conn, err := grpc.DialContext(ctx, host, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to server %s: %v", host, err)
	}

	client := NewModalClientClient(conn)
	return &Client{client: client}, nil
}

func (c *Client) LookupFunction(ctx context.Context, environment, app, name string) (*RemoteFunction, error) {
	resp, err := c.client.FunctionGet(ctx, &FunctionGetRequest{
		AppName:         app,
		ObjectTag:       name,
		Namespace:       DeploymentNamespace_DEPLOYMENT_NAMESPACE_WORKSPACE,
		EnvironmentName: environment,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get function: %v", err)
	}

	metadata := resp.HandleMetadata
	if metadata == nil {
		metadata = &FunctionHandleMetadata{}
	}

	if metadata.IsMethod {
		return nil, fmt.Errorf("method functions are not supported")
	}
	if metadata.FunctionType != Function_FUNCTION_TYPE_FUNCTION {
		return nil, fmt.Errorf("unsupported function type: %d", metadata.FunctionType)
	}

	return &RemoteFunction{resp.FunctionId, c.client}, nil
}
