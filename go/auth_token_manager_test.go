package modal

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type mockAuthClient struct {
	pb.ModalClientClient
	authToken    string
	authTokenErr error
	delay        time.Duration
	mu           sync.Mutex
	callCount    int
	lastOpts     []grpc.CallOption
}

func newMockAuthClient() *mockAuthClient {
	return &mockAuthClient{}
}

func (m *mockAuthClient) setAuthToken(token string) {
	m.mu.Lock()
	m.authToken = token
	m.mu.Unlock()
}

func (m *mockAuthClient) setAuthTokenError(err error) {
	m.mu.Lock()
	m.authTokenErr = err
	m.mu.Unlock()
}

func (m *mockAuthClient) setDelay(delay time.Duration) {
	m.mu.Lock()
	m.delay = delay
	m.mu.Unlock()
}

func (m *mockAuthClient) AuthTokenGet(ctx context.Context, req *pb.AuthTokenGetRequest, opts ...grpc.CallOption) (*pb.AuthTokenGetResponse, error) {
	m.mu.Lock()
	token := m.authToken
	err := m.authTokenErr
	delay := m.delay
	m.callCount++
	m.lastOpts = append([]grpc.CallOption(nil), opts...)
	m.mu.Unlock()

	select {
	case <-time.After(delay):
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if err != nil {
		return nil, err
	}
	return pb.AuthTokenGetResponse_builder{
		Token: token,
	}.Build(), nil
}

func (m *mockAuthClient) getCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

func (m *mockAuthClient) getLastOpts() []grpc.CallOption {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]grpc.CallOption(nil), m.lastOpts...)
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
	manager := newAuthTokenManager(mockClient, slog.Default())

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

	manager := newAuthTokenManager(mockClient, slog.Default())

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

func TestAuthTokenManager_FetchOnlyRetriesWithoutCachedToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	token := createTestJWT(time.Now().Unix() + 3600)
	mockClient.setAuthToken(token)
	manager := newAuthTokenManager(mockClient, slog.Default())

	_, err := manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	opts := mockClient.getLastOpts()
	var timeoutFound, retriesFound bool
	for _, opt := range opts {
		switch option := opt.(type) {
		case timeoutCallOption:
			timeoutFound = option.timeout == authTokenGetTimeout
		case retryCallOption:
			retriesFound = option.retries != nil && *option.retries == defaultRetryAttempts
		}
	}
	g.Expect(timeoutFound).Should(gomega.BeTrue())
	g.Expect(retriesFound).Should(gomega.BeTrue())

	manager.SetToken(createTestJWT(time.Now().Unix()-60), time.Now().Unix()-60)
	_, err = manager.FetchToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	opts = mockClient.getLastOpts()
	timeoutFound = false
	retriesFound = false
	for _, opt := range opts {
		switch option := opt.(type) {
		case timeoutCallOption:
			timeoutFound = option.timeout == authTokenGetTimeout
		case retryCallOption:
			retriesFound = option.retries != nil && *option.retries == 0
		}
	}
	g.Expect(timeoutFound).Should(gomega.BeTrue())
	g.Expect(retriesFound).Should(gomega.BeTrue())
}

func TestAuthTokenManager_IsExpired(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	manager := newAuthTokenManager(nil, slog.Default())

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

	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiringToken, now-60)
	mockClient.setAuthToken(freshToken)

	// GetToken should see the expired token and fetch a new one
	token, err := manager.GetToken(t.Context())
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(freshToken))
}

func TestAuthTokenManager_RefreshTokenPastRefreshPoint(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()

	// Token still valid for 60s, but already past its refresh point.
	expiringToken := createTestJWT(now + 60)
	freshToken := createTestJWT(now + 3600)

	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.setTokenWithRefreshAt(expiringToken, now+60, now-1)
	mockClient.setAuthToken(freshToken)

	// GetToken should proactively refresh
	token, err := manager.GetToken(t.Context())
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(freshToken))
}

func TestAuthTokenManager_ExpiredTokenIsDueForRefresh(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// An expiry in the past must not schedule the refresh point by a negative delay; the
	// token is simply due for refresh.
	now := time.Now().Unix()
	data := newTokenAndExpiry(createTestJWT(now-100), now-100)

	g.Expect(data.refreshAt).Should(gomega.Equal(now))
	g.Expect(needsRefresh(*data)).Should(gomega.BeTrue())
}

func TestAuthTokenManager_GetToken_EmptyResponse(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	// authToken is "" by default, so AuthTokenGet returns empty
	manager := newAuthTokenManager(mockClient, slog.Default())

	_, err := manager.GetToken(t.Context())
	g.Expect(err).Should(gomega.HaveOccurred())
}

func TestAuthTokenManager_ExpiredRefreshFailureBacksOff(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	mockClient.setAuthTokenError(fmt.Errorf("server blip"))
	now := time.Now().Unix()
	expiredToken := createTestJWT(now - 60)
	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiredToken, now-60)

	token, err := manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(expiredToken))

	token, err = manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(expiredToken))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
	g.Expect(manager.inBackoff()).Should(gomega.BeTrue())
}

func TestAuthTokenManager_NoCachedTokenRefreshFailureReturnsError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	mockClient.setAuthTokenError(fmt.Errorf("server blip"))
	manager := newAuthTokenManager(mockClient, slog.Default())

	_, err := manager.GetToken(t.Context())
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("server blip")))
}

func TestAuthTokenManager_EmptyResponseBacksOff(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()
	expiredToken := createTestJWT(now - 60)
	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiredToken, now-60)

	token, err := manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(expiredToken))

	token, err = manager.GetToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(expiredToken))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
	g.Expect(manager.inBackoff()).Should(gomega.BeTrue())
}

func TestAuthTokenManager_RefreshBackoffGrowsExponentially(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()
	expiredToken := createTestJWT(now - 60)
	freshToken := createTestJWT(now + 3600)
	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.SetToken(expiredToken, now-60)
	mockClient.setAuthTokenError(fmt.Errorf("server blip"))

	expectedBackoffs := []time.Duration{
		failureBackoffBase,
		time.Second,
		2 * time.Second,
		4 * time.Second,
		8 * time.Second,
		16 * time.Second,
		32 * time.Second,
		failureBackoffMax,
		failureBackoffMax,
	}
	for _, expected := range expectedBackoffs {
		before := time.Now().UnixMilli()
		manager.retryAfter.Store(nil)
		_, err := manager.FetchToken(t.Context())
		g.Expect(err).Should(gomega.HaveOccurred())
		ra := manager.retryAfter.Load()
		g.Expect(ra).ShouldNot(gomega.BeNil())
		g.Expect(ra.UnixMilli()).Should(gomega.BeNumerically(">=", before+expected.Milliseconds()))
		g.Expect(ra.UnixMilli()).Should(gomega.BeNumerically("<=", before+expected.Milliseconds()+1000))
	}

	mockClient.setAuthToken(freshToken)
	mockClient.setAuthTokenError(nil)
	manager.retryAfter.Store(nil)
	_, err := manager.FetchToken(t.Context())
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(manager.backoff.Load()).Should(gomega.Equal(failureBackoffBase))

	mockClient.setAuthTokenError(fmt.Errorf("server blip"))
	before := time.Now().UnixMilli()
	manager.retryAfter.Store(nil)
	_, err = manager.FetchToken(t.Context())
	g.Expect(err).Should(gomega.HaveOccurred())
	ra := manager.retryAfter.Load()
	g.Expect(ra).ShouldNot(gomega.BeNil())
	g.Expect(ra.UnixMilli()).Should(gomega.BeNumerically(">=", before+failureBackoffBase.Milliseconds()))
	g.Expect(ra.UnixMilli()).Should(gomega.BeNumerically("<=", before+failureBackoffBase.Milliseconds()+1000))
}

func TestAuthTokenManager_TimedOutFetchArmsBackoff(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	mockClient.setAuthToken(createTestJWT(time.Now().Unix() + 3600))
	mockClient.setDelay(time.Minute)

	manager := newAuthTokenManager(mockClient, slog.Default())

	// A server too slow to answer within the refresh timeout is a health signal like any other failure.
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	_, err := manager.FetchToken(ctx)

	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(manager.retryAfter.Load()).ShouldNot(gomega.BeNil())
	g.Expect(manager.inBackoff()).Should(gomega.BeTrue())
}

func TestAuthTokenManager_AuthDeniedDoesNotFallBack(t *testing.T) {
	t.Parallel()

	for _, code := range []codes.Code{codes.Unauthenticated, codes.PermissionDenied} {
		t.Run(code.String(), func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)

			mockClient := newMockAuthClient()
			now := time.Now().Unix()
			expiredToken := createTestJWT(now - 60)
			manager := newAuthTokenManager(mockClient, slog.Default())
			manager.SetToken(expiredToken, now-60)
			mockClient.setAuthTokenError(status.Error(code, "credentials rejected"))

			token, err := manager.GetToken(t.Context())
			g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("credentials rejected")))
			g.Expect(token).Should(gomega.BeEmpty())
			g.Expect(manager.GetCurrentToken()).Should(gomega.Equal(expiredToken))
			g.Expect(manager.retryAfter.Load()).Should(gomega.BeNil())
			g.Expect(manager.backoff.Load()).Should(gomega.Equal(failureBackoffBase))
			g.Expect(manager.inBackoff()).Should(gomega.BeFalse())
		})
	}
}

func TestAuthToken_ConcurrentGetTokenWithExpiredToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	now := time.Now().Unix()

	expiredToken := createTestJWT(now - 10)
	freshToken := createTestJWT(now + 7200)

	manager := newAuthTokenManager(mockClient, slog.Default())
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

func TestAuthToken_ConcurrentGetTokenWithFailingFetch(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockClient := newMockAuthClient()
	mockClient.setAuthTokenError(errors.New("auth server unavailable"))
	mockClient.setDelay(250 * time.Millisecond)

	manager := newAuthTokenManager(mockClient, slog.Default())

	var wg sync.WaitGroup
	errs := make([]error, 3)
	for i := range 3 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = manager.GetToken(t.Context())
		}(i)
	}
	wg.Wait()

	// Callers with no cached token share the failing refresh instead of each making their own request.
	g.Expect(errs).Should(gomega.HaveEach(gomega.HaveOccurred()))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
}

func TestAuthToken_NearExpiryRefreshDoesNotBlockOtherCallers(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	now := time.Now().Unix()
	nearExpiry := now + 290
	nearExpiryToken := createTestJWT(nearExpiry)
	freshToken := createTestJWT(now + 7200)

	mockClient := newMockAuthClient()
	mockClient.setAuthToken(freshToken)
	mockClient.setDelay(500 * time.Millisecond)

	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.setTokenWithRefreshAt(nearExpiryToken, nearExpiry, now-1)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = manager.GetToken(t.Context())
	}()
	time.Sleep(100 * time.Millisecond)

	// The token is still valid, so a caller arriving during the refresh gets it back without waiting for the request.
	start := time.Now()
	token, err := manager.GetToken(t.Context())
	elapsed := time.Since(start)
	wg.Wait()

	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(nearExpiryToken))
	g.Expect(elapsed).Should(gomega.BeNumerically("<", 100*time.Millisecond))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
}

func TestAuthToken_AbandonedNearExpiryRefreshDoesNotBlockOtherCallers(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	now := time.Now().Unix()
	nearExpiry := now + 290
	nearExpiryToken := createTestJWT(nearExpiry)

	mockClient := newMockAuthClient()
	mockClient.setAuthToken(createTestJWT(now + 7200))
	mockClient.setDelay(500 * time.Millisecond)

	manager := newAuthTokenManager(mockClient, slog.Default())
	manager.setTokenWithRefreshAt(nearExpiryToken, nearExpiry, now-1)

	// Start the refresh, then abandon it: the request keeps running without anyone waiting on it.
	ctx, cancel := context.WithCancel(t.Context())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = manager.GetToken(ctx)
	}()
	time.Sleep(100 * time.Millisecond)
	cancel()
	wg.Wait()

	// The token is still valid, so the next caller must not end up waiting on that abandoned request.
	start := time.Now()
	token, err := manager.GetToken(t.Context())
	elapsed := time.Since(start)

	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token).Should(gomega.Equal(nearExpiryToken))
	g.Expect(elapsed).Should(gomega.BeNumerically("<", 100*time.Millisecond))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
}

func TestAuthToken_CancellingCallerDoesNotCancelRefresh(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	token := createTestJWT(time.Now().Unix() + 3600)
	mockClient := newMockAuthClient()
	mockClient.setAuthToken(token)
	mockClient.setDelay(500 * time.Millisecond)

	manager := newAuthTokenManager(mockClient, slog.Default())

	ctx, cancel := context.WithCancel(t.Context())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = manager.GetToken(ctx)
	}()

	// Let the first caller start the refresh, then have a second caller join it before cancelling the first.
	time.Sleep(100 * time.Millisecond)
	var sharedToken string
	var sharedErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		sharedToken, sharedErr = manager.GetToken(t.Context())
	}()
	time.Sleep(100 * time.Millisecond)
	cancel()
	wg.Wait()

	// The refresh outlives the caller that started it, so the remaining caller still gets the token.
	g.Expect(sharedErr).ShouldNot(gomega.HaveOccurred())
	g.Expect(sharedToken).Should(gomega.Equal(token))
	g.Expect(mockClient.getCallCount()).Should(gomega.Equal(1))
}
