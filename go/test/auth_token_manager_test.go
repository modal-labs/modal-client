package test

import (
	"context"
	"log/slog"
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
	callCount int
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
	m.callCount++
	m.mu.Unlock()

	return pb.AuthTokenGetResponse_builder{
		Token: token,
	}.Build(), nil
}

func (m *mockAuthClient) getCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

func createTestJWT(expiry int64) string {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"exp": expiry,
		"iat": time.Now().Unix(),
	})

	tokenString, _ := token.SignedString([]byte("walter-test"))
	return tokenString
}

func TestAuthTokenManager_DecodeJWT(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	manager := modal.NewAuthTokenManager(mockClient, slog.Default())

	validToken := createTestJWT(123456789)
	mockClient.setAuthToken(validToken)

	_, err := manager.FetchToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(manager.GetCurrentToken()).Should(gomega.Equal(validToken))
}

func TestAuthTokenManager_LazyFetch(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	token := createTestJWT(time.Now().Unix() + 3600)
	mockClient.setAuthToken(token)

	manager := modal.NewAuthTokenManager(mockClient, slog.Default())

	// First GetToken lazily fetches
	firstToken, err := manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(firstToken).Should(gomega.Equal(token))

	// Second GetToken returns cached
	secondToken, err := manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(secondToken).Should(gomega.Equal(token))

	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
}

func TestAuthTokenManager_IsExpired(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	manager := modal.NewAuthTokenManager(nil, slog.Default())

	manager.SetToken("token", time.Now().Unix()+3600)
	g.Expect(manager.IsExpired()).Should(gomega.BeFalse())

	manager.SetToken("token", time.Now().Unix()-3600)
	g.Expect(manager.IsExpired()).Should(gomega.BeTrue())
}

func TestAuthTokenManager_RefreshExpiredToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()

	expiringToken := createTestJWT(now - 60)
	freshToken := createTestJWT(now + 3600)

	manager := modal.NewAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiringToken, now-60)
	mockClient.setAuthToken(freshToken)

	// GetToken should see the expired token and fetch a new one
	token, err := manager.GetToken(t.Context())
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(freshToken))
}

func TestAuthTokenManager_RefreshNearExpiryToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()

	// Token within RefreshWindow of expiry (60s left, window is 300s)
	expiringToken := createTestJWT(now + 60)
	freshToken := createTestJWT(now + 3600)

	manager := modal.NewAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiringToken, now+60)
	mockClient.setAuthToken(freshToken)

	// GetToken should proactively refresh
	token, err := manager.GetToken(t.Context())
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(freshToken))
}

func TestAuthTokenManager_GetToken_EmptyResponse(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	// authToken is "" by default, so AuthTokenGet returns empty
	manager := modal.NewAuthTokenManager(mockClient, slog.Default())

	_, err := manager.GetToken(t.Context())
	g.Expect(err).Should(gomega.HaveOccurred())
}

func TestAuthToken_ConcurrentGetTokenWithExpiredToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()

	expiredToken := createTestJWT(now - 10)
	freshToken := createTestJWT(now + 7200)

	manager := modal.NewAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiredToken, now-10)
	mockClient.setAuthToken(freshToken)

	var wg sync.WaitGroup
	results := make([]string, 3)
	for i := range 3 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			token, err := manager.GetToken(t.Context())
			g.Expect(err).ShouldNot(gomega.HaveOccurred())
			results[idx] = token
		}(i)
	}
	wg.Wait()

	g.Expect(results).Should(gomega.HaveEach(freshToken))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
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
