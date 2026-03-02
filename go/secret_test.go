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
	ctx := t.Context()

	mockClient := &Client{Secrets: &mockSecretService{}}

	env := map[string]string{"B": "2", "C": "3"}
	existingSecret := &Secret{SecretID: "st-existing"}
	secrets := []*Secret{existingSecret}

	result, err := mergeEnvIntoSecrets(ctx, mockClient, &env, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(2))
	g.Expect(result[0]).To(gomega.Equal(existingSecret))
	g.Expect(result[1].SecretID).To(gomega.Equal("st-mock-env"))
}

func TestMergeEnvIntoSecrets_WithOnlyEnv(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mockClient := &Client{Secrets: &mockSecretService{}}

	env := map[string]string{"B": "2", "C": "3"}

	result, err := mergeEnvIntoSecrets(ctx, mockClient, &env, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(1))
	g.Expect(result[0].SecretID).To(gomega.Equal("st-mock-env"))
}

func TestMergeEnvIntoSecrets_WithEmptyEnvReturnsExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	existingSecret := &Secret{SecretID: "st-existing"}
	env := map[string]string{}
	secrets := []*Secret{existingSecret}

	result, err := mergeEnvIntoSecrets(ctx, &Client{}, &env, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(1))
	g.Expect(result[0]).To(gomega.Equal(existingSecret))
}

func TestMergeEnvIntoSecrets_WithNilEnvReturnsExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	existingSecret := &Secret{SecretID: "st-existing"}
	secrets := []*Secret{existingSecret}

	result, err := mergeEnvIntoSecrets(ctx, &Client{}, nil, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(1))
	g.Expect(result[0]).To(gomega.Equal(existingSecret))
}

func TestMergeEnvIntoSecrets_WithOnlyExistingSecrets(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	secret1 := &Secret{SecretID: "st-secret1"}
	secret2 := &Secret{SecretID: "st-secret2"}
	secrets := []*Secret{secret1, secret2}

	result, err := mergeEnvIntoSecrets(ctx, &Client{}, nil, &secrets)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(2))
	g.Expect(result[0]).To(gomega.Equal(secret1))
	g.Expect(result[1]).To(gomega.Equal(secret2))
}

func TestMergeEnvIntoSecrets_WithNoEnvAndNoSecretsReturnsEmptyArray(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	result, err := mergeEnvIntoSecrets(ctx, &Client{}, nil, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).To(gomega.HaveLen(0))
	g.Expect(result).To(gomega.BeNil())
}
