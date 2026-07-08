package modal

import (
	"context"
	"fmt"
	"regexp"
	"sync"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// envVarNameRegex matches valid environment variable names: letters, numbers,
// and underscores, not starting with a number. This mirrors the server-side
// validation so invalid keys are rejected before any request is made.
var envVarNameRegex = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

// validateEnvVarName returns an InvalidError if key is not a valid environment
// variable name. It mirrors the server-side validation so invalid keys are
// rejected before any request is made.
func validateEnvVarName(key string) error {
	if !envVarNameRegex.MatchString(key) {
		return InvalidError{Exception: fmt.Sprintf("Secret key name %q is invalid for environment variables. Only letters, numbers, and underscores are allowed, and the name cannot start with a number.", key)}
	}
	return nil
}

// SecretService provides Secret related operations.
type SecretService interface {
	FromName(ctx context.Context, name string, params *SecretFromNameParams) (*Secret, error)
	FromMap(ctx context.Context, keyValuePairs map[string]string, params *SecretFromMapParams) (*Secret, error)
	Delete(ctx context.Context, name string, params *SecretDeleteParams) error
}

type secretServiceImpl struct{ client *Client }

// secretHydrator resolves a lazy Secret to a server-side SecretID. Each
// construction path (FromName, FromMap, ...) supplies its own implementation
// carrying whatever inputs that path needs.
type secretHydrator interface {
	hydrate(ctx context.Context, client *Client) (secretID string, err error)
}

// Secret represents a Modal Secret.
type Secret struct {
	SecretID string
	Name     string

	// hydrator resolves SecretID lazily. It is nil for Secrets constructed
	// already-hydrated, and is not consulted again once SecretID is set.
	hydrator secretHydrator
	// hydrateMu serializes hydrate so the SecretID is resolved at most once even
	// if multiple goroutines hydrate the same Secret concurrently. A failed
	// attempt is not cached, so callers may retry after a transient error.
	hydrateMu sync.Mutex
}

// hydrate lazily resolves the server-side SecretID via the Secret's hydrator. It
// is a no-op for Secrets that already have a SecretID (e.g. those from FromName,
// or an already-hydrated lazy Secret).
func (s *Secret) hydrate(ctx context.Context, client *Client) error {
	if s.hydrator == nil {
		return nil // not a lazy Secret; SecretID already set
	}
	s.hydrateMu.Lock()
	defer s.hydrateMu.Unlock()
	if s.SecretID != "" {
		return nil // already hydrated by an earlier call
	}

	id, err := s.hydrator.hydrate(ctx, client)
	if err != nil {
		return err
	}

	s.SecretID = id
	return nil
}

// secretFromMapHydrator creates an ephemeral server-side Secret from a locally
// provided env map. Used by Secrets created via FromMap.
type secretFromMapHydrator struct {
	envDict     map[string]string
	environment string
}

// secretEnvDictHydrator reports whether secret is a non-nil env-dict Secret and,
// if so, returns its hydrator. This is the worker fast path: such Secrets can be
// passed to the worker as environment variables without a SecretGetOrCreate.
func secretEnvDictHydrator(secret *Secret) (*secretFromMapHydrator, bool) {
	if secret == nil {
		return nil, false
	}
	h, ok := secret.hydrator.(*secretFromMapHydrator)
	return h, ok
}

func (h *secretFromMapHydrator) hydrate(ctx context.Context, client *Client) (string, error) {
	resp, err := client.cpClient.SecretGetOrCreate(ctx, pb.SecretGetOrCreateRequest_builder{
		ObjectCreationType: pb.ObjectCreationType_OBJECT_CREATION_TYPE_EPHEMERAL,
		EnvDict:            h.envDict,
		EnvironmentName:    h.environment,
	}.Build())
	if err != nil {
		return "", err
	}

	id := resp.GetSecretId()
	client.logger.DebugContext(ctx, "Created ephemeral Secret", "secret_id", id)
	return id, nil
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
		EnvironmentName: firstNonEmpty(params.Environment, s.client.profile.Environment),
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

	envDict := make(map[string]string, len(keyValuePairs))
	for k, v := range keyValuePairs {
		if err := validateEnvVarName(k); err != nil {
			return nil, err
		}
		envDict[k] = v
	}
	environment := firstNonEmpty(params.Environment, s.client.profile.Environment)

	// The Secret is lazy: hydrate creates the ephemeral server-side Secret from
	// envDict the first time the SecretID is needed, at most once even under
	// concurrent callers.
	return &Secret{hydrator: &secretFromMapHydrator{envDict: envDict, environment: environment}}, nil
}

// hydrateSecrets hydrates each Secret in the slice so that its SecretID is
// available. Returns an error if any Secret is nil or cannot be hydrated.
func hydrateSecrets(ctx context.Context, client *Client, secrets []*Secret) error {
	for i, secret := range secrets {
		if secret == nil {
			return InvalidError{Exception: fmt.Sprintf("secret at index %d must not be nil", i)}
		}
	}
	g, ctx := errgroup.WithContext(ctx)
	for _, secret := range secrets {
		if secret.hydrator == nil {
			continue // already hydrated (e.g. from FromName)
		}
		g.Go(func() error {
			return secret.hydrate(ctx, client)
		})
	}
	return g.Wait()
}

// splitEnvDictAndResolvableSecrets partitions secrets into a merged env dict
// (from Secrets created locally via FromMap) and the remaining "resolvable"
// Secrets that must be hydrated to a SecretID before use (e.g. from FromName).
//
// Locally-created Secrets can be passed directly to the worker as environment
// variables, avoiding a SecretGetOrCreate round-trip. This function does not
// validate its input: nil Secrets are placed in the resolvable list rather than
// dropped, leaving it to hydrateSecrets to reject them with an error.
func splitEnvDictAndResolvableSecrets(secrets []*Secret) (map[string]string, []*Secret) {
	envDict := map[string]string{}
	var resolvable []*Secret
	for _, secret := range secrets {
		if h, ok := secretEnvDictHydrator(secret); ok {
			for k, v := range h.envDict {
				envDict[k] = v
			}
		} else {
			resolvable = append(resolvable, secret)
		}
	}
	return envDict, resolvable
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
// If env contains values, it creates a new (lazy) Secret from the env map and
// appends it to the existing secrets. The appended Secret is hydrated together
// with the others when its SecretID is needed.
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
