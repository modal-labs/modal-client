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
