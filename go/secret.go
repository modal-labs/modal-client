package modal

import (
	"context"
	"fmt"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SecretService provides Secret related operations.
type SecretService interface {
	FromName(ctx context.Context, name string, params *SecretFromNameParams) (*Secret, error)
	FromMap(ctx context.Context, keyValuePairs map[string]string, params *SecretFromMapParams) (*Secret, error)
	Delete(ctx context.Context, name string, params *SecretDeleteParams) error
}

type secretServiceImpl struct{ client *Client }

// Secret represents a Modal Secret.
type Secret struct {
	SecretID string
	Name     string
}

// SecretFromNameParams are options for finding Modal Secrets.
type SecretFromNameParams struct {
	Environment  string
	RequiredKeys []string
}

// FromName references a Secret by its name.
func (s *secretServiceImpl) FromName(ctx context.Context, name string, params *SecretFromNameParams) (*Secret, error) {
	if params == nil {
		params = &SecretFromNameParams{}
	}

	resp, err := s.client.cpClient.SecretGetOrCreate(ctx, pb.SecretGetOrCreateRequest_builder{
		DeploymentName:  name,
		EnvironmentName: environmentName(params.Environment, s.client.profile),
		RequiredKeys:    params.RequiredKeys,
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Secret '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Retrieved Secret", "secret_id", resp.GetSecretId(), "secret_name", name)
	return &Secret{SecretID: resp.GetSecretId(), Name: name}, nil
}

// SecretFromMapParams are options for creating a Secret from a key/value map.
type SecretFromMapParams struct {
	Environment string
}

// SecretDeleteParams are options for client.Secrets.Delete.
type SecretDeleteParams struct {
	Environment  string
	AllowMissing bool
}

// FromMap creates a Secret from a map of key-value pairs.
func (s *secretServiceImpl) FromMap(ctx context.Context, keyValuePairs map[string]string, params *SecretFromMapParams) (*Secret, error) {
	if params == nil {
		params = &SecretFromMapParams{}
	}

	resp, err := s.client.cpClient.SecretGetOrCreate(ctx, pb.SecretGetOrCreateRequest_builder{
		ObjectCreationType: pb.ObjectCreationType_OBJECT_CREATION_TYPE_EPHEMERAL,
		EnvDict:            keyValuePairs,
		EnvironmentName:    environmentName(params.Environment, s.client.profile),
	}.Build())
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Created ephemeral Secret", "secret_id", resp.GetSecretId())
	return &Secret{SecretID: resp.GetSecretId()}, nil
}

// Delete deletes a named Secret.
//
// Warning: Deletion is irreversible and will affect any Apps currently using the Secret.
func (s *secretServiceImpl) Delete(ctx context.Context, name string, params *SecretDeleteParams) error {
	if params == nil {
		params = &SecretDeleteParams{}
	}

	secret, err := s.FromName(ctx, name, &SecretFromNameParams{
		Environment: params.Environment,
	})

	if err != nil {
		if _, ok := err.(NotFoundError); ok && params.AllowMissing {
			return nil
		}
		return err
	}

	_, err = s.client.cpClient.SecretDelete(ctx, pb.SecretDeleteRequest_builder{
		SecretId: secret.SecretID,
	}.Build())

	if err != nil {
		if st, ok := status.FromError(err); ok && st.Code() == codes.NotFound && params.AllowMissing {
			return nil
		}
		return err
	}

	s.client.logger.DebugContext(ctx, "Deleted Secret", "secret_name", name, "secret_id", secret.SecretID)
	return nil
}

// mergeEnvIntoSecrets merges environment variables into the secrets list.
// If env contains values, it creates a new Secret from the env map and appends it to the existing secrets.
func mergeEnvIntoSecrets(ctx context.Context, client *Client, env *map[string]string, secrets *[]*Secret) ([]*Secret, error) {
	var result []*Secret
	if secrets != nil {
		result = append(result, *secrets...)
	}

	if env != nil && len(*env) > 0 {
		envSecret, err := client.Secrets.FromMap(ctx, *env, nil)
		if err != nil {
			return nil, err
		}
		result = append(result, envSecret)
	}

	return result, nil
}
