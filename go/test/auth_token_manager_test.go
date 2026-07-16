package test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	modal "github.com/modal-labs/modal-client/go"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
)

type mockAuthClient struct {
	pb.ModalClientClient
	authToken string
	mu        sync.Mutex
}

func newMockAuthClient() *mockAuthClient {
	return &mockAuthClient{}
}

func (m *mockAuthClient) setAuthToken(token string) {
	m.mu.Lock()
	m.authToken = token
	m.mu.Unlock()
}

func (m *mockAuthClient) AuthTokenGet(ctx context.Context, req *pb.AuthTokenGetRequest, opts ...grpc.CallOption) (*pb.AuthTokenGetResponse, error) {
	m.mu.Lock()
	token := m.authToken
	m.mu.Unlock()

	return pb.AuthTokenGetResponse_builder{
		Token: token,
	}.Build(), nil
}

func createTestJWT(expiry int64) string {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"exp": expiry,
		"iat": time.Now().Unix(),
	})

	tokenString, _ := token.SignedString([]byte("walter-test"))
	return tokenString
}

func TestClient_Close(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	token := createTestJWT(time.Now().Unix() + 3600)
	mockClient.setAuthToken(token)

	client, err := modal.NewClientWithOptions(&modal.ClientParams{
		ControlPlaneClient: mockClient,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	client.Close()
}

func TestClient_MultipleInstancesSeparateManagers(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient1 := newMockAuthClient()
	token1 := createTestJWT(time.Now().Unix() + 3600)
	mockClient1.setAuthToken(token1)

	mockClient2 := newMockAuthClient()
	token2 := createTestJWT(time.Now().Unix() + 3600)
	mockClient2.setAuthToken(token2)

	client1, err1 := modal.NewClientWithOptions(&modal.ClientParams{
		ControlPlaneClient: mockClient1,
	})
	g.Expect(err1).ShouldNot(gomega.HaveOccurred())
	defer client1.Close()

	client2, err2 := modal.NewClientWithOptions(&modal.ClientParams{
		ControlPlaneClient: mockClient2,
	})
	g.Expect(err2).ShouldNot(gomega.HaveOccurred())
	defer client2.Close()
}
