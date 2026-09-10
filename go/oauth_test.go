package modal

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"math/big"
	"strings"
	"testing"

	"github.com/golang-jwt/jwt/v5"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/metadata"
)

func rsaPrivateKeyPEM(t *testing.T, bits int) (string, *rsa.PrivateKey) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("MarshalPKCS8PrivateKey: %v", err)
	}
	return string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})), key
}

func TestParseOAuthJWTKey(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	pemPKCS8, want := rsaPrivateKeyPEM(t, 2048)

	got, err := parseOAuthJWTKey(pemPKCS8)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(got.Equal(want)).To(gomega.BeTrue())

	got, err = parseOAuthJWTKey(strings.ReplaceAll(pemPKCS8, "\n", "\\n"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(got.N.Cmp(want.N)).To(gomega.Equal(0))
}

func TestParseOAuthJWTKeyRejectsInvalid(t *testing.T) {
	t.Parallel()

	shortPEM, _ := rsaPrivateKeyPEM(t, 1024)
	pub, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	pubDER, err := x509.MarshalPKIXPublicKey(&pub.PublicKey)
	if err != nil {
		t.Fatalf("MarshalPKIXPublicKey: %v", err)
	}
	publicPEM := string(pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: pubDER}))

	cases := []struct {
		name    string
		pem     string
		message string
	}{
		{name: "garbage", pem: "not a private key", message: "unencrypted RSA private key encoded as PEM"},
		{name: "too small", pem: shortPEM, message: "at least 2048 bits"},
		{name: "public key", pem: publicPEM, message: "unencrypted RSA private key encoded as PEM"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			_, err := parseOAuthJWTKey(tc.pem)
			g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring(tc.message)))
		})
	}
}

func TestMintOAuthClientAssertion(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, key := rsaPrivateKeyPEM(t, 2048)
	kid := rsaJWKThumbprint(&key.PublicKey)

	first, err := mintOAuthClientAssertion("oc-client-id", key)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	second, err := mintOAuthClientAssertion("oc-client-id", key)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(first).NotTo(gomega.Equal(second))

	token, err := jwt.Parse(first, func(token *jwt.Token) (any, error) {
		return &key.PublicKey, nil
	}, jwt.WithValidMethods([]string{jwt.SigningMethodRS256.Alg()}), jwt.WithAudience(oauthClientAssertionAudience))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(token.Header["kid"]).To(gomega.Equal(kid))
	g.Expect(token.Header["alg"]).To(gomega.Equal("RS256"))

	claims, ok := token.Claims.(jwt.MapClaims)
	g.Expect(ok).To(gomega.BeTrue())
	g.Expect(claims["iss"]).To(gomega.Equal("oc-client-id"))
	g.Expect(claims["sub"]).To(gomega.Equal("oc-client-id"))
}

func TestRSAJWKThumbprintMatchesRFC7638(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// RFC 7638 §3.1 example JWK (n/e only; alg and kid are excluded from the thumbprint).
	n, err := base64.RawURLEncoding.DecodeString("0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	e, err := base64.RawURLEncoding.DecodeString("AQAB")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	pub := &rsa.PublicKey{
		N: new(big.Int).SetBytes(n),
		E: int(new(big.Int).SetBytes(e).Int64()),
	}
	g.Expect(rsaJWKThumbprint(pub)).To(gomega.Equal("NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs"))
}

func TestInjectRequiredHeadersWithOAuthJWTKey(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	pemBytes, key := rsaPrivateKeyPEM(t, 2048)

	c := &Client{
		profile: Profile{
			OAuthRefreshToken: "refresh-token",
			OAuthClientID:     "oc-client-id",
			OAuthJWTKey:       pemBytes,
		},
		sdkVersion:  "test-version",
		oauthJWTKey: key,
	}
	ctx, err := injectRequiredHeaders(context.Background(), c)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	md, ok := metadata.FromOutgoingContext(ctx)
	g.Expect(ok).To(gomega.BeTrue())
	g.Expect(md.Get("x-modal-refresh-token")).To(gomega.Equal([]string{"refresh-token"}))
	g.Expect(md.Get("x-modal-oauth-client-id")).To(gomega.Equal([]string{"oc-client-id"}))
	g.Expect(md.Get("x-modal-oauth-client-assertion")).NotTo(gomega.BeEmpty())
	g.Expect(md.Get("x-modal-oauth-client-secret")).To(gomega.BeEmpty())
}

func TestNewClientWithOptionsRejectsInvalidOAuthJWTKey(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := NewClientWithOptions(&ClientParams{
		OAuthCredentials: &OAuthCredentialsParams{
			RefreshToken: "refresh-token",
			ClientID:     "oc-client-id",
			JWTKey:       "not a private key",
		},
	})
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("unencrypted RSA private key encoded as PEM")))
}
