package modal

import (
	"context"
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type imageBuildMockClient struct {
	pb.ModalClientClient
	imageGetOrCreateCalls int
}

//nolint:staticcheck // name must match the generated ModalClientClient interface method.
func (m *imageBuildMockClient) ImageFromId(context.Context, *pb.ImageFromIdRequest, ...grpc.CallOption) (*pb.ImageFromIdResponse, error) {
	return pb.ImageFromIdResponse_builder{ImageId: "im-base"}.Build(), nil
}

func (m *imageBuildMockClient) ImageGetOrCreate(context.Context, *pb.ImageGetOrCreateRequest, ...grpc.CallOption) (*pb.ImageGetOrCreateResponse, error) {
	m.imageGetOrCreateCalls++
	return nil, status.Error(codes.NotFound, "No Image with ID 'im-missing' found")
}

func TestImageBuildReturnsNotFoundErrorForMissingBaseImage(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	mock := &imageBuildMockClient{}
	client := newEnvironmentServiceTestClient(t, mock)
	client.profile.ImageBuilderVersion = "2025.06"

	baseImage, err := client.Images.FromID(t.Context(), "im-base", nil)
	g.Expect(err).ShouldNot(HaveOccurred())
	image := baseImage.DockerfileCommands([]string{"RUN echo layer"}, nil)
	_, err = image.Build(t.Context(), &App{AppID: "ap-test", Environment: "dev"}, nil)

	g.Expect(err).To(Equal(NotFoundError{"No Image with ID 'im-missing' found"}))
	g.Expect(mock.imageGetOrCreateCalls).To(Equal(1))
}

type registryAuthMockClient struct {
	pb.ModalClientClient
	keys             []string
	registryAuthType pb.RegistryAuthType
}

func (m *registryAuthMockClient) SecretGetInfo(context.Context, *pb.SecretGetInfoRequest, ...grpc.CallOption) (*pb.SecretGetInfoResponse, error) {
	return pb.SecretGetInfoResponse_builder{Metadata: pb.SecretMetadata_builder{Keys: m.keys}.Build()}.Build(), nil
}

func (m *registryAuthMockClient) ImageGetOrCreate(_ context.Context, req *pb.ImageGetOrCreateRequest, _ ...grpc.CallOption) (*pb.ImageGetOrCreateResponse, error) {
	m.registryAuthType = req.GetImage().GetImageRegistryConfig().GetRegistryAuthType()
	return pb.ImageGetOrCreateResponse_builder{
		ImageId: "im-registry",
		Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
	}.Build(), nil
}

func buildRegistryImage(t *testing.T, mock *registryAuthMockClient, secret *Secret) error {
	client := newEnvironmentServiceTestClient(t, mock)
	client.profile.ImageBuilderVersion = "2025.06"
	image := client.Images.FromRegistry("ghcr.io/my-org/my-image:latest", &ImageFromRegistryParams{Secret: secret})
	_, err := image.Build(t.Context(), &App{AppID: "ap-test", Environment: "dev"}, nil)
	return err
}

func TestImageFromRegistryInfersRegistryAuthType(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		keys []string
		want pb.RegistryAuthType
	}{
		{[]string{"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"}, pb.RegistryAuthType_REGISTRY_AUTH_TYPE_AWS},
		{[]string{"AWS_ROLE_ARN", "AWS_REGION"}, pb.RegistryAuthType_REGISTRY_AUTH_TYPE_AWS},
		{[]string{"SERVICE_ACCOUNT_JSON"}, pb.RegistryAuthType_REGISTRY_AUTH_TYPE_GCP},
		{[]string{"REGISTRY_USERNAME", "REGISTRY_PASSWORD", "OTHER"}, pb.RegistryAuthType_REGISTRY_AUTH_TYPE_STATIC_CREDS},
	} {
		g := NewWithT(t)
		mock := &registryAuthMockClient{keys: tc.keys}
		g.Expect(buildRegistryImage(t, mock, &Secret{SecretID: "sc-registry"})).To(Succeed())
		g.Expect(mock.registryAuthType).To(Equal(tc.want))
	}
}

func TestImageFromRegistryRefusesUnresolvableSecret(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		keys    []string
		message string
	}{
		{[]string{"TOKEN"}, "must include"},
		{[]string{"SERVICE_ACCOUNT_JSON", "AWS_ROLE_ARN", "AWS_REGION"}, "more than one"},
	} {
		g := NewWithT(t)
		err := buildRegistryImage(t, &registryAuthMockClient{}, &Secret{SecretID: "sc-registry", keys: tc.keys})
		g.Expect(err).To(BeAssignableToTypeOf(InvalidError{}))
		g.Expect(err.Error()).To(ContainSubstring(tc.message))
	}
}

type imagePlatformMockClient struct {
	pb.ModalClientClient
	platforms []string
}

func (m *imagePlatformMockClient) ImageGetOrCreate(_ context.Context, req *pb.ImageGetOrCreateRequest, _ ...grpc.CallOption) (*pb.ImageGetOrCreateResponse, error) {
	m.platforms = append(m.platforms, req.GetImage().GetPlatform())
	return pb.ImageGetOrCreateResponse_builder{
		ImageId: "im-layer",
		Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
	}.Build(), nil
}

func TestImageBuildSetsPlatformOnRootLayerOnly(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	mock := &imagePlatformMockClient{}
	client := newEnvironmentServiceTestClient(t, mock)
	client.profile.ImageBuilderVersion = "2025.06"

	image := client.Images.FromRegistry("alpine:3.21", &ImageFromRegistryParams{Platform: "linux/arm64"}).
		DockerfileCommands([]string{"RUN echo layer"}, nil)
	_, err := image.Build(t.Context(), &App{AppID: "ap-test", Environment: "dev"}, nil)

	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(mock.platforms).To(Equal([]string{"linux/arm64", ""}))
}
