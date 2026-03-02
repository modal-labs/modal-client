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
	g.Expect(secret.SecretID).Should(gomega.HavePrefix("st-"))

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Secrets: []*modal.Secret{secret}, Command: []string{"printenv", "key"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("value\n"))
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
