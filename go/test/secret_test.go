package test

import (
	"io"
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestSecretFromName(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	secret, err := tc.Secrets.FromName(ctx, "libmodal-test-secret", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(secret.SecretID).Should(gomega.HavePrefix("st-"))
	g.Expect(secret.Name).To(gomega.Equal("libmodal-test-secret"))

	_, err = tc.Secrets.FromName(ctx, "missing-secret", nil)
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("Secret 'missing-secret' not found")))
}

func TestSecretFromNameWithRequiredKeys(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	secret, err := tc.Secrets.FromName(ctx, "libmodal-test-secret", &modal.SecretFromNameParams{
		RequiredKeys: []string{"a", "b", "c"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(secret.SecretID).Should(gomega.HavePrefix("st-"))

	_, err = tc.Secrets.FromName(ctx, "libmodal-test-secret", &modal.SecretFromNameParams{
		RequiredKeys: []string{"a", "b", "c", "missing-key"},
	})
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("Secret is missing key(s): missing-key")))
}

func TestSecretFromMap(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	secret, err := tc.Secrets.FromMap(ctx, map[string]string{"key": "value"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	// FromMap is lazy: no Secret is created on the server until it is used.
	g.Expect(secret.SecretID).Should(gomega.BeEmpty())

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Secrets: []*modal.Secret{secret}, Command: []string{"printenv", "key"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	// Using the Secret in a Sandbox hydrates it into a server-side Secret.
	g.Expect(secret.SecretID).Should(gomega.HavePrefix("st-"))

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("value\n"))
}

// registerSandboxCreateDeps registers mock handlers for the RPCs that a Sandbox
// create issues before the SandboxCreate/SandboxCreateV2 call: app lookup,
// image builder version, and image build.
func registerSandboxCreateDeps(mock *grpcmock.MockClient) {
	grpcmock.HandleUnary(mock, "AppGetOrCreate",
		func(req *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
			return pb.AppGetOrCreateResponse_builder{AppId: "ap-123"}.Build(), nil
		},
	)
	grpcmock.HandleUnary(mock, "EnvironmentGetOrCreate",
		func(req *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
			return pb.EnvironmentGetOrCreateResponse_builder{
				EnvironmentId: "en-test",
				Metadata: pb.EnvironmentMetadata_builder{
					Name:     "test",
					Settings: pb.EnvironmentSettings_builder{ImageBuilderVersion: "2025.06"}.Build(),
				}.Build(),
			}.Build(), nil
		},
	)
	grpcmock.HandleUnary(mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-123",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)
}

func TestSecretFromMapIsLazy(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mock := newGRPCMockClient(t)

	// No RPC handlers are registered: FromMap must not make any control-plane
	// call, otherwise the mock would error on an unexpected RPC.
	secret, err := mock.Secrets.FromMap(t.Context(), map[string]string{"key": "value"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(secret.SecretID).Should(gomega.BeEmpty())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxCreateHydratesFromMapSecret(t *testing.T) {
	// Unset MODAL_IMAGE_BUILDER_VERSION so the build resolves it via EnvironmentGetOrCreate.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)
	registerSandboxCreateDeps(mock)

	grpcmock.HandleUnary(mock, "SecretGetOrCreate",
		func(req *pb.SecretGetOrCreateRequest) (*pb.SecretGetOrCreateResponse, error) {
			// The lazy FromMap Secret is hydrated as an ephemeral Secret on create.
			g.Expect(req.GetObjectCreationType()).To(gomega.Equal(pb.ObjectCreationType_OBJECT_CREATION_TYPE_EPHEMERAL))
			g.Expect(req.GetEnvDict()).To(gomega.Equal(map[string]string{"FOO": "bar"}))
			return pb.SecretGetOrCreateResponse_builder{SecretId: "st-ephemeral"}.Build(), nil
		},
	)
	grpcmock.HandleUnary(mock, "SandboxCreate",
		func(req *pb.SandboxCreateRequest) (*pb.SandboxCreateResponse, error) {
			g.Expect(req.GetDefinition().GetSecretIds()).To(gomega.ContainElement("st-ephemeral"))
			return pb.SandboxCreateResponse_builder{SandboxId: validV1SandboxID}.Build(), nil
		},
	)

	app, err := mock.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	image := mock.Images.FromRegistry("alpine:3.21", nil)

	secret, err := mock.Secrets.FromMap(ctx, map[string]string{"FOO": "bar"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb, err := mock.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Secrets: []*modal.Secret{secret}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb.SandboxID).To(gomega.Equal(validV1SandboxID))

	// The Secret was hydrated in place.
	g.Expect(secret.SecretID).To(gomega.Equal("st-ephemeral"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxExperimentalCreatePassesEnvAsEphemeralSecrets(t *testing.T) {
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)
	registerSandboxCreateDeps(mock)

	// Note: no SecretGetOrCreate handler is registered. The V2 path must pass
	// env vars via ephemeral_secrets rather than creating a Secret for them, so
	// no SecretGetOrCreate RPC should occur.
	grpcmock.HandleUnary(mock, "SandboxCreateV2",
		func(req *pb.SandboxCreateV2Request) (*pb.SandboxCreateV2Response, error) {
			g.Expect(req.GetEphemeralSecrets().GetContents()).To(gomega.Equal(map[string]string{"FOO": "bar"}))
			g.Expect(req.GetDefinition().GetSecretIds()).To(gomega.BeEmpty())
			return pb.SandboxCreateV2Response_builder{SandboxId: validV2SandboxID}.Build(), nil
		},
	)

	app, err := mock.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	image := mock.Images.FromRegistry("alpine:3.21", nil)

	sb, err := mock.Sandboxes.ExperimentalCreate(ctx, app, image, &modal.SandboxCreateParams{
		Env: map[string]string{"FOO": "bar"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb.SandboxID).To(gomega.Equal(validV2SandboxID))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxExperimentalCreatePassesFromMapSecretAsEphemeralSecrets(t *testing.T) {
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)
	registerSandboxCreateDeps(mock)

	// No SecretGetOrCreate handler is registered. Locally-created FromMap Secrets
	// must be folded into ephemeral_secrets in the V2 path rather than hydrated
	// into a server-side Secret, so no SecretGetOrCreate RPC should occur.
	grpcmock.HandleUnary(mock, "SandboxCreateV2",
		func(req *pb.SandboxCreateV2Request) (*pb.SandboxCreateV2Response, error) {
			// params.Env takes precedence over the FromMap value on key collisions.
			g.Expect(req.GetEphemeralSecrets().GetContents()).To(gomega.Equal(map[string]string{"FOO": "from-env", "BAZ": "qux"}))
			g.Expect(req.GetDefinition().GetSecretIds()).To(gomega.BeEmpty())
			return pb.SandboxCreateV2Response_builder{SandboxId: validV2SandboxID}.Build(), nil
		},
	)

	app, err := mock.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	image := mock.Images.FromRegistry("alpine:3.21", nil)

	secret, err := mock.Secrets.FromMap(ctx, map[string]string{"FOO": "from-secret", "BAZ": "qux"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb, err := mock.Sandboxes.ExperimentalCreate(ctx, app, image, &modal.SandboxCreateParams{
		Secrets: []*modal.Secret{secret},
		Env:     map[string]string{"FOO": "from-env"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb.SandboxID).To(gomega.Equal(validV2SandboxID))

	// The FromMap Secret was never hydrated into a server-side Secret.
	g.Expect(secret.SecretID).Should(gomega.BeEmpty())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxExperimentalCreateRejectsInvalidEnvVarName(t *testing.T) {
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)
	registerSandboxCreateDeps(mock)

	// No SandboxCreateV2 handler is registered: an invalid env var name must be
	// rejected locally before any create RPC is made.
	app, err := mock.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	image := mock.Images.FromRegistry("alpine:3.21", nil)

	_, err = mock.Sandboxes.ExperimentalCreate(ctx, app, image, &modal.SandboxCreateParams{
		Env: map[string]string{"1INVALID": "value"},
	})
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("is invalid for environment variables")))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSecretDeleteSuccess(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/SecretGetOrCreate",
		func(req *pb.SecretGetOrCreateRequest) (*pb.SecretGetOrCreateResponse, error) {
			return pb.SecretGetOrCreateResponse_builder{
				SecretId: "st-test-123",
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "/SecretDelete",
		func(req *pb.SecretDeleteRequest) (*emptypb.Empty, error) {
			g.Expect(req.GetSecretId()).To(gomega.Equal("st-test-123"))
			return &emptypb.Empty{}, nil
		},
	)

	err := mock.Secrets.Delete(ctx, "test-secret", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSecretDeleteWithAllowMissing(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/SecretGetOrCreate",
		func(req *pb.SecretGetOrCreateRequest) (*pb.SecretGetOrCreateResponse, error) {
			return nil, modal.NotFoundError{Exception: "Secret 'missing' not found"}
		},
	)

	err := mock.Secrets.Delete(ctx, "missing", &modal.SecretDeleteParams{
		AllowMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSecretDeleteWithAllowMissingDeleteRPCNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "/SecretGetOrCreate",
		func(req *pb.SecretGetOrCreateRequest) (*pb.SecretGetOrCreateResponse, error) {
			return pb.SecretGetOrCreateResponse_builder{SecretId: "st-test-123"}.Build(), nil
		},
	)

	grpcmock.HandleUnary(mock, "/SecretDelete",
		func(req *pb.SecretDeleteRequest) (*emptypb.Empty, error) {
			return nil, status.Errorf(codes.NotFound, "Secret not found")
		},
	)

	err := mock.Secrets.Delete(ctx, "test-secret", &modal.SecretDeleteParams{AllowMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSecretDeleteWithAllowMissingFalseThrows(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/SecretGetOrCreate",
		func(req *pb.SecretGetOrCreateRequest) (*pb.SecretGetOrCreateResponse, error) {
			return nil, modal.NotFoundError{Exception: "Secret 'missing' not found"}
		},
	)

	err := mock.Secrets.Delete(ctx, "missing", &modal.SecretDeleteParams{
		AllowMissing: false,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	var notFoundErr modal.NotFoundError
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(notFoundErr))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
