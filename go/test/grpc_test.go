package test

import (
	"context"
	"net"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

type slowModalServer struct {
	pb.UnimplementedModalClientServer
	sleepDuration time.Duration
}

// AppGetOrCreate is just chosen arbitrarily as a GRPC method to use for testing.
func (s *slowModalServer) AppGetOrCreate(ctx context.Context, req *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
	select {
	case <-time.After(s.sleepDuration):
		return pb.AppGetOrCreateResponse_builder{AppId: req.GetAppName()}.Build(), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *slowModalServer) AuthTokenGet(ctx context.Context, req *pb.AuthTokenGetRequest) (*pb.AuthTokenGetResponse, error) {
	// Mock JWT with "x" mock header, base64 enc of {"exp":9999999999}, and "x" mock signature,
	// since AuthTokenManager.FetchToken() warns if the JWT doesn't have an "exp" field.
	const mockJWT = "x.eyJleHAiOjk5OTk5OTk5OTl9.x"
	return pb.AuthTokenGetResponse_builder{Token: mockJWT}.Build(), nil
}

func TestClientRespectsContextDeadline(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		serverSleep    time.Duration
		contextTimeout time.Duration
		expectTimeout  bool
	}{
		{
			name:           "deadline exceeded",
			serverSleep:    100 * time.Millisecond,
			contextTimeout: 10 * time.Millisecond,
			expectTimeout:  true,
		},
		{
			name:           "completes before deadline",
			serverSleep:    10 * time.Millisecond,
			contextTimeout: 100 * time.Millisecond,
			expectTimeout:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewWithT(t)

			lis := bufconn.Listen(1024 * 1024)

			grpcServer := grpc.NewServer()
			pb.RegisterModalClientServer(grpcServer, &slowModalServer{
				sleepDuration: tc.serverSleep,
			})

			go func() {
				if err := grpcServer.Serve(lis); err != nil {
					t.Logf("Server error: %v", err)
				}
			}()
			defer grpcServer.Stop()

			bufDialer := func(context.Context, string) (net.Conn, error) {
				return lis.Dial()
			}

			conn, err := grpc.NewClient("passthrough:///bufnet",
				grpc.WithContextDialer(bufDialer),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			g.Expect(err).ShouldNot(gomega.HaveOccurred())
			defer func() {
				if conn.GetState() != connectivity.Shutdown {
					if err := conn.Close(); err != nil {
						t.Errorf("failed to close gRPC connection: %v", err)
					}
				}
			}()

			client, err := modal.NewClientWithOptions(&modal.ClientParams{
				TokenID:            "test-token-id",
				TokenSecret:        "test-token-secret",
				Environment:        "test",
				ControlPlaneClient: pb.NewModalClientClient(conn),
				ControlPlaneConn:   conn,
			})
			g.Expect(err).ShouldNot(gomega.HaveOccurred())
			defer client.Close()

			ctxWithTimeout, cancel := context.WithTimeout(t.Context(), tc.contextTimeout)
			defer cancel()

			app, err := client.Apps.FromName(ctxWithTimeout, "test-app", nil)

			if tc.expectTimeout {
				g.Expect(err).Should(gomega.HaveOccurred())
				st, ok := status.FromError(err)
				g.Expect(ok).To(gomega.BeTrue())
				g.Expect(st.Code()).To(gomega.Equal(codes.DeadlineExceeded))
			} else {
				g.Expect(err).ShouldNot(gomega.HaveOccurred())
				g.Expect(app.AppID).To(gomega.Equal("test-app"))
			}
		})
	}
}
