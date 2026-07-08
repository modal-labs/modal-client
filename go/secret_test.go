package modal

import (
	"context"
	"testing"

	"github.com/onsi/gomega"
)

type mockSecretService struct{ SecretService }

func (m *mockSecretService) FromMap(ctx context.Context, keyValuePairs map[string]string, params *SecretFromMapParams) (*Secret, error) {
	return &Secret{SecretID: "st-mock-env"}, nil
}

func TestMergeEnvIntoSecrets_WithEnvAndExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)

	mockClient := &Client{Secrets: &mockSecretService{}}

	env := map[string]string{"B": "2", "C": "3"}
	existingSecret := &Secret{SecretID: "st-existing"}
	secrets := []*Secret{existingSecret}

	result, err := mergeEnvIntoSecrets(t.Context(), mockClient, &env, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(2))
	g.Expect(result[0]).To(gomega.Equal(existingSecret))
	g.Expect(result[1].SecretID).To(gomega.Equal("st-mock-env"))
}

func TestMergeEnvIntoSecrets_WithOnlyEnv(t *testing.T) {
	g := gomega.NewWithT(t)

	mockClient := &Client{Secrets: &mockSecretService{}}

	env := map[string]string{"B": "2", "C": "3"}

	result, err := mergeEnvIntoSecrets(t.Context(), mockClient, &env, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(1))
	g.Expect(result[0].SecretID).To(gomega.Equal("st-mock-env"))
}

func TestMergeEnvIntoSecrets_WithEmptyEnvReturnsExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)

	existingSecret := &Secret{SecretID: "st-existing"}
	env := map[string]string{}
	secrets := []*Secret{existingSecret}

	result, err := mergeEnvIntoSecrets(t.Context(), &Client{}, &env, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(1))
	g.Expect(result[0]).To(gomega.Equal(existingSecret))
}

func TestMergeEnvIntoSecrets_WithNilEnvReturnsExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)

	existingSecret := &Secret{SecretID: "st-existing"}
	secrets := []*Secret{existingSecret}

	result, err := mergeEnvIntoSecrets(t.Context(), &Client{}, nil, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(1))
	g.Expect(result[0]).To(gomega.Equal(existingSecret))
}

func TestMergeEnvIntoSecrets_WithOnlyExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)

	secret1 := &Secret{SecretID: "st-secret1"}
	secret2 := &Secret{SecretID: "st-secret2"}
	secrets := []*Secret{secret1, secret2}

	result, err := mergeEnvIntoSecrets(t.Context(), &Client{}, nil, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(2))
	g.Expect(result[0]).To(gomega.Equal(secret1))
	g.Expect(result[1]).To(gomega.Equal(secret2))
}

func TestMergeEnvIntoSecrets_WithNoEnvAndNoSecretsReturnsEmptyArray(t *testing.T) {
	g := gomega.NewWithT(t)

	result, err := mergeEnvIntoSecrets(t.Context(), &Client{}, nil, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(0))
	g.Expect(result).To(gomega.BeNil())
}

func TestFromMapIsLazy(t *testing.T) {
	g := gomega.NewWithT(t)

	svc := &secretServiceImpl{client: &Client{profile: Profile{Environment: "my-env"}}}

	input := map[string]string{"A": "1", "B": "2"}
	secret, err := svc.FromMap(t.Context(), input, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	// Lazy: no SecretID is allocated and a hydrator is set up for later.
	g.Expect(secret.SecretID).To(gomega.BeEmpty())
	h, ok := secretEnvDictHydrator(secret)
	g.Expect(ok).To(gomega.BeTrue())
	g.Expect(h.envDict).To(gomega.Equal(map[string]string{"A": "1", "B": "2"}))
}

func TestFromMapRejectsInvalidEnvVarNames(t *testing.T) {
	g := gomega.NewWithT(t)

	svc := &secretServiceImpl{client: &Client{profile: Profile{Environment: "my-env"}}}

	for _, key := range []string{"", "1FOO", "FO-O", "FO O", "FOO=", "foo.bar"} {
		secret, err := svc.FromMap(t.Context(), map[string]string{key: "value"}, nil)
		g.Expect(secret).To(gomega.BeNil())
		g.Expect(err).To(gomega.BeAssignableToTypeOf(InvalidError{}))
	}

	// Valid names (letters, digits, underscores; not starting with a digit) are accepted.
	for _, key := range []string{"FOO", "_foo", "foo_BAR_123", "a"} {
		_, err := svc.FromMap(t.Context(), map[string]string{key: "value"}, nil)
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
	}
}

func TestSplitEnvDictAndResolvableSecrets(t *testing.T) {
	g := gomega.NewWithT(t)

	local1 := &Secret{hydrator: &secretFromMapHydrator{envDict: map[string]string{"A": "1", "B": "2"}}}
	local2 := &Secret{hydrator: &secretFromMapHydrator{envDict: map[string]string{"B": "override", "C": "3"}}}
	named := &Secret{SecretID: "st-named"}

	// Local Secrets are merged in slice order (so local2's B wins); named and
	// nil Secrets are kept in the resolvable list.
	envDict, resolvable := splitEnvDictAndResolvableSecrets([]*Secret{local1, named, local2, nil})

	g.Expect(envDict).To(gomega.Equal(map[string]string{"A": "1", "B": "override", "C": "3"}))
	g.Expect(resolvable).To(gomega.Equal([]*Secret{named, nil}))
}

func TestSplitEnvDictAndResolvableSecretsNoLocalSecrets(t *testing.T) {
	g := gomega.NewWithT(t)

	named := &Secret{SecretID: "st-named"}
	envDict, resolvable := splitEnvDictAndResolvableSecrets([]*Secret{named})

	g.Expect(envDict).To(gomega.BeEmpty())
	g.Expect(resolvable).To(gomega.Equal([]*Secret{named}))
}

func TestHydrateSecretsRejectsNil(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	err := hydrateSecrets(ctx, &Client{}, []*Secret{{SecretID: "st-1"}, nil})
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("secret at index 1 must not be nil")))
}

func TestHydrateSecretsSkipsAlreadyHydrated(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	// Secrets that already have a SecretID have a nil hydrate closure, so no RPC
	// is attempted (the nil cpClient would panic if one were).
	err := hydrateSecrets(ctx, &Client{}, []*Secret{{SecretID: "st-1"}, {SecretID: "st-2"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
}

func TestHydrateSandboxSecretsDoesNotMutateCallerSlice(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	// A slice with spare capacity would let a naive append() write the mount
	// Secret into the caller's backing array.
	backing := make([]*Secret, 1, 4)
	backing[0] = &Secret{SecretID: "st-1"}
	mounts := map[string]*CloudBucketMount{"/mnt": {Secret: &Secret{SecretID: "st-mount"}}}

	err := hydrateSandboxSecrets(ctx, &Client{}, backing, mounts)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	// The mount Secret must not have leaked into the caller's backing array.
	g.Expect(backing[:cap(backing)][1]).To(gomega.BeNil())
}
