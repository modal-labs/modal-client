package modal

import (
	"context"
	"encoding/base64"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
)

type mockEnvironmentClient struct {
	pb.ModalClientClient
	EnvironmentGetOrCreateCallCount atomic.Uint64
	Envs                            sync.Map
}

func (m *mockEnvironmentClient) EnvironmentGetOrCreate(ctx context.Context, req *pb.EnvironmentGetOrCreateRequest, opts ...grpc.CallOption) (*pb.EnvironmentGetOrCreateResponse, error) {
	m.EnvironmentGetOrCreateCallCount.Add(1)
	name := req.GetDeploymentName()
	if resp, ok := m.Envs.Load(name); ok {
		return resp.(*pb.EnvironmentGetOrCreateResponse), nil
	}
	return pb.EnvironmentGetOrCreateResponse_builder{
		EnvironmentId: "en-custom-id",
		Metadata: pb.EnvironmentMetadata_builder{
			Name: name,
			Settings: pb.EnvironmentSettings_builder{
				ImageBuilderVersion: "2025.06",
				WebhookSuffix:       "",
			}.Build(),
		}.Build(),
	}.Build(), nil
}

func (m *mockEnvironmentClient) AuthTokenGet(ctx context.Context, req *pb.AuthTokenGetRequest, opts ...grpc.CallOption) (*pb.AuthTokenGetResponse, error) {
	expMsg := []byte("{'exp': 9999999999}")
	encodedString := base64.URLEncoding.EncodeToString(expMsg)
	mockJWT := fmt.Sprintf("x.%s.x", encodedString)
	return pb.AuthTokenGetResponse_builder{Token: mockJWT}.Build(), nil
}

func TestGetEnvironmentLocalConfig(t *testing.T) {
	g := NewWithT(t)
	ctx := t.Context()

	t.Setenv("MODAL_PROFILE", "")
	// Unset `MODAL_IMAGE_BUILDER_VERSION` so that this test uses the image builder version from `EnvironmentGetOrCreate`.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")

	mockClient := &mockEnvironmentClient{}
	mockClient.Envs.Store("", pb.EnvironmentGetOrCreateResponse_builder{
		EnvironmentId: "en-custom-id",
		Metadata: pb.EnvironmentMetadata_builder{
			Settings: pb.EnvironmentSettings_builder{
				ImageBuilderVersion: "2024.10",
			}.Build(),
		}.Build()}.Build())

	client, err := NewClientWithOptions(&ClientParams{
		ControlPlaneClient: mockClient,
		Config: &config{
			"default": rawProfile{
				ImageBuilderVersion: "2024.04",
				Active:              true,
			},
		},
	})
	g.Expect(err).ShouldNot(HaveOccurred())
	defer client.Close()

	version, err := client.getImageBuilderVersion(ctx, "")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(version).Should(Equal("2024.04"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(0))
}

func TestGetEnvironment(t *testing.T) {
	g := NewWithT(t)
	ctx := t.Context()

	// Unset `MODAL_IMAGE_BUILDER_VERSION` so that this test uses the image builder version from `EnvironmentGetOrCreate`.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")

	mockClient := &mockEnvironmentClient{}
	mockClient.Envs.Store("", pb.EnvironmentGetOrCreateResponse_builder{
		EnvironmentId: "en-main-123",
		Metadata: pb.EnvironmentMetadata_builder{
			Name: "main",
			Settings: pb.EnvironmentSettings_builder{
				ImageBuilderVersion: "2024.10",
				WebhookSuffix:       "modal.run",
			}.Build(),
		}.Build()}.Build())

	mockClient.Envs.Store("dev", pb.EnvironmentGetOrCreateResponse_builder{
		EnvironmentId: "en-dev-321",
		Metadata: pb.EnvironmentMetadata_builder{
			Name: "dev",
			Settings: pb.EnvironmentSettings_builder{
				ImageBuilderVersion: "2025.06",
				WebhookSuffix:       "modal.dev.run",
			}.Build(),
		}.Build(),
	}.Build())

	client, err := NewClientWithOptions(&ClientParams{ControlPlaneClient: mockClient})
	g.Expect(err).ShouldNot(HaveOccurred())
	defer client.Close()

	env, err := client.environmentManager.fetchEnvironment(ctx, "")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(env.ID).Should(Equal("en-main-123"))
	g.Expect(env.Name).Should(Equal("main"))
	g.Expect(env.Settings.ImageBuilderVersion).Should(Equal("2024.10"))
	g.Expect(env.Settings.WebhookSuffix).Should(Equal("modal.run"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(1))

	version, err := client.environmentManager.GetImageBuilderVersion(ctx, "")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(version).Should(Equal("2024.10"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(1))

	// Used cached value
	env2, err := client.environmentManager.fetchEnvironment(ctx, "")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(env2).Should(BeIdenticalTo(env))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(1))

	version, err = client.environmentManager.GetImageBuilderVersion(ctx, "")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(version).Should(Equal("2024.10"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(1))

	// New fetch from the server
	envDev, err := client.environmentManager.fetchEnvironment(ctx, "dev")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(envDev.ID).Should(Equal("en-dev-321"))
	g.Expect(envDev.Name).Should(Equal("dev"))
	g.Expect(envDev.Settings.ImageBuilderVersion).Should(Equal("2025.06"))
	g.Expect(envDev.Settings.WebhookSuffix).Should(Equal("modal.dev.run"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(2))

	version, err = client.environmentManager.GetImageBuilderVersion(ctx, "dev")
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(version).Should(Equal("2025.06"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(2))

}

func TestGetImageBuilderVersionWithEnvironmentOverride(t *testing.T) {
	g := NewWithT(t)
	ctx := t.Context()

	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")

	mockClient := &mockEnvironmentClient{}
	mockClient.Envs.Store("", pb.EnvironmentGetOrCreateResponse_builder{
		EnvironmentId: "en-main-123",
		Metadata: pb.EnvironmentMetadata_builder{
			Name: "main",
			Settings: pb.EnvironmentSettings_builder{
				ImageBuilderVersion: "2024.10",
			}.Build(),
		}.Build(),
	}.Build())
	mockClient.Envs.Store("dev", pb.EnvironmentGetOrCreateResponse_builder{
		EnvironmentId: "en-dev-321",
		Metadata: pb.EnvironmentMetadata_builder{
			Name: "dev",
			Settings: pb.EnvironmentSettings_builder{
				ImageBuilderVersion: "2025.06",
			}.Build(),
		}.Build(),
	}.Build())

	client, err := NewClientWithOptions(&ClientParams{ControlPlaneClient: mockClient})
	g.Expect(err).ShouldNot(HaveOccurred())
	defer client.Close()

	// App created with a per-call environment override, profile default is "".
	app := &App{AppID: "ap-123", Name: "my-app", Environment: "dev"}

	version, err := client.getImageBuilderVersion(ctx, app.Environment)
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(version).Should(Equal("2025.06"))
	g.Expect(int(mockClient.EnvironmentGetOrCreateCallCount.Load())).Should(Equal(1))
}
