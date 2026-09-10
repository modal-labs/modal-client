package modal

// Private-key JWT (RFC 7523) assertions for OAuth client authentication.

import (
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"math/big"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

const (
	oauthClientAssertionAudience        = "modal-server"
	oauthClientAssertionLifetimeSeconds = 5 * 60
	minRSAKeySize                       = 2048
)

func parseOAuthJWTKey(privateKeyPEM string) (*rsa.PrivateKey, error) {
	privateKey, err := jwt.ParseRSAPrivateKeyFromPEM([]byte(strings.ReplaceAll(privateKeyPEM, "\\n", "\n")))
	if err != nil {
		return nil, InvalidError{Exception: "oauth_jwt_key must be an unencrypted RSA private key encoded as PEM."}
	}
	if privateKey.N.BitLen() < minRSAKeySize {
		return nil, InvalidError{Exception: "oauth_jwt_key must be at least 2048 bits."}
	}
	return privateKey, nil
}

func rsaJWKThumbprint(pub *rsa.PublicKey) string {
	e := base64.RawURLEncoding.EncodeToString(big.NewInt(int64(pub.E)).Bytes())
	n := base64.RawURLEncoding.EncodeToString(pub.N.Bytes())
	sum := sha256.Sum256([]byte(fmt.Sprintf(`{"e":%q,"kty":"RSA","n":%q}`, e, n)))
	return base64.RawURLEncoding.EncodeToString(sum[:])
}

func mintOAuthClientAssertion(clientID string, key *rsa.PrivateKey) (string, error) {
	now := time.Now()
	token := jwt.NewWithClaims(jwt.SigningMethodRS256, jwt.MapClaims{
		"iss": clientID,
		"sub": clientID,
		"aud": oauthClientAssertionAudience,
		"iat": now.Unix(),
		"exp": now.Add(oauthClientAssertionLifetimeSeconds * time.Second).Unix(),
		"jti": uuid.NewString(),
	})
	token.Header["kid"] = rsaJWKThumbprint(&key.PublicKey)
	assertion, err := token.SignedString(key)
	if err != nil {
		return "", fmt.Errorf("sign OAuth client assertion: %w", err)
	}
	return assertion, nil
}
