package modal

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	// Start refreshing this many seconds before the token expires
	RefreshWindow       = 5 * 60
	authTokenGetTimeout = 5 * time.Second
	// After a failed refresh, wait before hitting the server again, growing exponentially
	// with consecutive failures between these bounds.
	failureBackoffBase = 500 * time.Millisecond
	failureBackoffMax  = 60 * time.Second
	// If the token doesn't have an expiry field, default to current time plus this value (not expected).
	DefaultExpiryOffset = 20 * 60
)

type tokenAndExpiry struct {
	token  string
	expiry int64
}

type atomicDuration struct {
	nanos atomic.Int64
}

func (d *atomicDuration) Load() time.Duration {
	return time.Duration(d.nanos.Load())
}

func (d *atomicDuration) Store(v time.Duration) {
	d.nanos.Store(int64(v))
}

// authTokenManager manages authentication tokens, refreshing them lazily
// when GetToken is called. Tokens are refreshed when expired or within
// RefreshWindow seconds of expiry.
type authTokenManager struct {
	client pb.ModalClientClient
	logger *slog.Logger

	tokenAndExpiry atomic.Pointer[tokenAndExpiry]
	refreshMu      sync.Mutex
	retryAfter     atomic.Pointer[time.Time]
	backoff        atomicDuration
}

func newAuthTokenManager(client pb.ModalClientClient, logger *slog.Logger) *authTokenManager {
	manager := &authTokenManager{
		client: client,
		logger: logger,
	}

	manager.tokenAndExpiry.Store(&tokenAndExpiry{
		token:  "",
		expiry: 0,
	})
	manager.backoff.Store(failureBackoffBase)

	return manager
}

// GetToken returns a valid auth token, fetching or refreshing as needed.
//
// Three states:
//  1. Valid token (not near expiry): returned immediately, no locking.
//  2. No token or expired: all callers block until a fresh token is fetched.
//     Only one goroutine makes the RPC; others wait on the mutex then see the
//     new token via a double-check.
//  3. Valid but within RefreshWindow of expiry: one goroutine refreshes
//     (blocking only itself); concurrent callers get the old, still-valid token.
func (m *authTokenManager) GetToken(ctx context.Context) (string, error) {
	data := m.tokenAndExpiry.Load()

	if data.token == "" || isExpired(*data) {
		return m.tryRefreshToken(ctx)
	}

	if m.shouldRefresh(*data) && m.refreshMu.TryLock() {
		defer m.refreshMu.Unlock()
		token, err := m.FetchToken(ctx)
		if err != nil {
			if isAuthDenied(err) {
				return "", err
			}
			m.logger.WarnContext(ctx, "Auth token refresh failed; falling back to cached token", "error", err)
			return data.token, nil
		}
		return token, nil
	}

	return data.token, nil
}

func (m *authTokenManager) tryRefreshToken(ctx context.Context) (string, error) {
	token, err := m.lockedRefreshToken(ctx)
	if err == nil {
		return token, nil
	}
	if isAuthDenied(err) {
		return "", err
	}
	data := m.tokenAndExpiry.Load()
	if data.token == "" {
		return "", err
	}
	m.logger.WarnContext(ctx, "Auth token refresh failed; falling back to cached token", "error", err)
	return data.token, nil
}

// lockedRefreshToken blocks until the mutex is acquired, then refreshes if still needed.
// Returns the current valid token.
func (m *authTokenManager) lockedRefreshToken(ctx context.Context) (string, error) {
	m.refreshMu.Lock()
	defer m.refreshMu.Unlock()

	data := m.tokenAndExpiry.Load()
	if data.token != "" && !m.shouldRefresh(*data) {
		return data.token, nil
	}
	return m.FetchToken(ctx)
}

// FetchToken fetches a new token using AuthTokenGet() and stores it.
func (m *authTokenManager) FetchToken(ctx context.Context) (string, error) {
	retries := 0
	if m.tokenAndExpiry.Load().token == "" {
		// No cached token to fall back on, so a failure is user-visible: retry transient errors.
		// Otherwise one attempt, and retryAfter handles the cooldown.
		retries = defaultRetryAttempts
	}
	resp, err := m.client.AuthTokenGet(ctx, &pb.AuthTokenGetRequest{},
		timeoutCallOption{timeout: authTokenGetTimeout},
		retryCallOption{retries: &retries},
	)
	if err == nil && resp.GetToken() == "" {
		err = fmt.Errorf("internal error: did not receive auth token from server, please contact Modal support")
	}
	if err != nil {
		if isAuthDenied(err) {
			return "", err
		}
		// A caller-cancelled context (or a caller deadline shorter than our own
		// timeout) isn't a server-health signal, so don't arm the shared cooldown.
		if ctx.Err() != nil {
			return "", err
		}
		// Back off (exponentially on consecutive failures) so we don't hammer a struggling server.
		cur := m.backoff.Load()
		t := time.Now().Add(cur)
		m.retryAfter.Store(&t)
		next := cur * 2
		if next > failureBackoffMax {
			next = failureBackoffMax
		}
		m.backoff.Store(next)
		return "", err
	}

	token := resp.GetToken()
	var expiry int64
	if exp := m.decodeJWT(token); exp > 0 {
		expiry = exp
	} else {
		m.logger.WarnContext(ctx, "x-modal-auth-token does not contain exp field")
		// We'll use the token, and set the expiry to 20 min from now.
		expiry = time.Now().Unix() + DefaultExpiryOffset
	}

	m.tokenAndExpiry.Store(&tokenAndExpiry{
		token:  token,
		expiry: expiry,
	})

	timeUntilRefresh := time.Duration(expiry-time.Now().Unix()-RefreshWindow) * time.Second
	m.logger.DebugContext(ctx, "Fetched auth token",
		"expires_in", time.Until(time.Unix(expiry, 0)),
		"refresh_in", timeUntilRefresh)

	m.retryAfter.Store(nil)
	m.backoff.Store(failureBackoffBase)
	return token, nil
}

// Extracts the exp claim from a JWT token.
func (m *authTokenManager) decodeJWT(token string) int64 {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return 0
	}

	payload := parts[1]
	for len(payload)%4 != 0 {
		payload += "="
	}

	decoded, err := base64.URLEncoding.DecodeString(payload)
	if err != nil {
		return 0
	}

	var claims map[string]interface{}
	if err := json.Unmarshal(decoded, &claims); err != nil {
		return 0
	}

	if exp, ok := claims["exp"].(float64); ok {
		return int64(exp)
	}

	return 0
}

// GetCurrentToken returns the current cached token.
func (m *authTokenManager) GetCurrentToken() string {
	return m.tokenAndExpiry.Load().token
}

// IsExpired checks if the current token is expired.
func (m *authTokenManager) IsExpired() bool {
	return isExpired(*m.tokenAndExpiry.Load())
}

func isExpired(data tokenAndExpiry) bool {
	return time.Now().Unix() >= data.expiry
}

func needsRefresh(data tokenAndExpiry) bool {
	return time.Now().Unix() >= data.expiry-RefreshWindow
}

func (m *authTokenManager) inBackoff() bool {
	retryAfter := m.retryAfter.Load()
	return retryAfter != nil && time.Now().Before(*retryAfter)
}

func (m *authTokenManager) shouldRefresh(data tokenAndExpiry) bool {
	return needsRefresh(data) && !m.inBackoff()
}

func isAuthDenied(err error) bool {
	// The credentials themselves were rejected (e.g. revoked/invalid key), as opposed to a
	// transient outage/overload. Fail fast instead of reusing the cached token or backing off.
	st, ok := status.FromError(err)
	return ok && (st.Code() == codes.Unauthenticated || st.Code() == codes.PermissionDenied)
}

// SetToken sets the token and expiry (for testing).
func (m *authTokenManager) SetToken(token string, expiry int64) {
	m.tokenAndExpiry.Store(&tokenAndExpiry{
		token:  token,
		expiry: expiry,
	})
}
