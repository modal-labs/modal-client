package modal

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync/atomic"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/singleflight"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	// Start refreshing this many seconds before the token expires
	RefreshWindow       = 5 * 60
	authTokenGetTimeout = 5 * time.Second
	// Bounds a whole shared refresh, whose retries are detached from any caller's context and so otherwise unlimited.
	// Deliberately shorter than the full retry budget of authTokenGetTimeout per attempt: a refresh that has been
	// failing this long is better given up on, and callers with a cached token keep using it. Matches the Python
	// client's AUTH_TOKEN_RETRY_TOTAL_TIMEOUT.
	authTokenRefreshTimeout = 3 * authTokenGetTimeout
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
	refreshGroup   singleflight.Group
	// Claimed by the caller that takes on a pre-expiry refresh, so the others keep using the still-valid token
	// instead of waiting for the request.
	refreshing atomic.Bool
	retryAfter atomic.Pointer[time.Time]
	backoff    atomicDuration
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
//     Only one goroutine makes the RPC; the others wait for its outcome.
//  3. Valid but within RefreshWindow of expiry: one goroutine refreshes
//     (blocking only itself); concurrent callers get the old, still-valid token.
func (m *authTokenManager) GetToken(ctx context.Context) (string, error) {
	data := m.tokenAndExpiry.Load()

	if data.token == "" || isExpired(*data) {
		return m.tryRefreshToken(ctx)
	}

	if m.shouldRefresh(*data) && m.refreshing.CompareAndSwap(false, true) {
		results := m.startSharedRefresh(ctx)
		var token string
		var err error
		select {
		case result := <-results:
			m.refreshing.Store(false)
			token, err = tokenFromRefresh(result)
		case <-ctx.Done():
			// The refresh outlives this caller, so keep it claimed until it finishes: later callers should keep using
			// the still-valid token instead of waiting on a request they don't need.
			go func() {
				<-results
				m.refreshing.Store(false)
			}()
			err = ctx.Err()
		}
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
	token, err := m.sharedRefreshToken(ctx)
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

// sharedRefreshToken returns the token from a single in-flight refresh: the first caller makes the RPC while the
// others wait for its outcome, so a burst of callers results in one AuthTokenGet whether it succeeds or fails. The
// request is detached from the calling context, so a caller going away doesn't abort the refresh the remaining
// callers are waiting on, and bounded by authTokenRefreshTimeout so a stuck refresh can't block later ones.
func (m *authTokenManager) sharedRefreshToken(ctx context.Context) (string, error) {
	results := m.startSharedRefresh(ctx)
	select {
	case result := <-results:
		return tokenFromRefresh(result)
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// startSharedRefresh joins the in-flight refresh, starting one if there isn't one, and returns the channel its
// outcome is delivered on.
func (m *authTokenManager) startSharedRefresh(ctx context.Context) <-chan singleflight.Result {
	return m.refreshGroup.DoChan("refresh", func() (any, error) {
		// Maybe another goroutine refreshed while this call was being shared.
		if data := m.tokenAndExpiry.Load(); data.token != "" && !m.shouldRefresh(*data) {
			return data.token, nil
		}
		rpcCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), authTokenRefreshTimeout)
		defer cancel()
		return m.FetchToken(rpcCtx)
	})
}

func tokenFromRefresh(result singleflight.Result) (string, error) {
	if result.Err != nil {
		return "", result.Err
	}
	// singleflight predates generics, so the token comes back as an any.
	return result.Val.(string), nil
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
