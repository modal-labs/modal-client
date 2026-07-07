package modal

import (
	"context"
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc"
)

func TestBuildFunctionOptionsProto_NilOptions(t *testing.T) {
	g := gomega.NewWithT(t)

	options, err := buildFunctionOptionsProto(nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(options).Should(gomega.BeNil())
}

func TestBuildFunctionOptionsProto_WithCPUAndCPULimit(t *testing.T) {
	g := gomega.NewWithT(t)

	cpu := 2.0
	cpuLimit := 4.5
	options, err := buildFunctionOptionsProto(&functionOptions{
		cpu:      &cpu,
		cpuLimit: &cpuLimit,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(options).ShouldNot(gomega.BeNil())

	resources := options.GetResources()
	g.Expect(resources.GetMilliCpu()).To(gomega.Equal(uint32(2000)))
	g.Expect(resources.GetMilliCpuMax()).To(gomega.Equal(uint32(4500)))
}

func TestBuildFunctionOptionsProto_CPULimitLowerThanCPU(t *testing.T) {
	g := gomega.NewWithT(t)

	cpu := 4.0
	cpuLimit := 2.0
	_, err := buildFunctionOptionsProto(&functionOptions{
		cpu:      &cpu,
		cpuLimit: &cpuLimit,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("the CPU request (4.000000) cannot be higher than CPULimit (2.000000)"))
}

func TestBuildFunctionOptionsProto_CPULimitWithoutCPU(t *testing.T) {
	g := gomega.NewWithT(t)

	cpuLimit := 4.0
	_, err := buildFunctionOptionsProto(&functionOptions{
		cpuLimit: &cpuLimit,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must also specify non-zero CPU request when CPULimit is specified"))
}

func TestBuildFunctionOptionsProto_WithMemoryAndMemoryLimit(t *testing.T) {
	g := gomega.NewWithT(t)

	memoryMiB := 1024
	memoryLimitMiB := 2048
	options, err := buildFunctionOptionsProto(&functionOptions{
		memoryMiB:      &memoryMiB,
		memoryLimitMiB: &memoryLimitMiB,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(options).ShouldNot(gomega.BeNil())

	resources := options.GetResources()
	g.Expect(resources.GetMemoryMb()).To(gomega.Equal(uint32(1024)))
	g.Expect(resources.GetMemoryMbMax()).To(gomega.Equal(uint32(2048)))
}

func TestBuildFunctionOptionsProto_MemoryLimitLowerThanMemory(t *testing.T) {
	g := gomega.NewWithT(t)

	memoryMiB := 2048
	memoryLimitMiB := 1024
	_, err := buildFunctionOptionsProto(&functionOptions{
		memoryMiB:      &memoryMiB,
		memoryLimitMiB: &memoryLimitMiB,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("the MemoryMiB request (2048) cannot be higher than MemoryLimitMiB (1024)"))
}

func TestBuildFunctionOptionsProto_MemoryLimitWithoutMemory(t *testing.T) {
	g := gomega.NewWithT(t)

	memoryLimitMiB := 2048
	_, err := buildFunctionOptionsProto(&functionOptions{
		memoryLimitMiB: &memoryLimitMiB,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must also specify non-zero MemoryMiB request when MemoryLimitMiB is specified"))
}

func TestBuildFunctionOptionsProto_NegativeCPU(t *testing.T) {
	g := gomega.NewWithT(t)

	cpu := -1.0
	_, err := buildFunctionOptionsProto(&functionOptions{
		cpu: &cpu,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be a positive number"))
}

func TestBuildFunctionOptionsProto_ZeroCPU(t *testing.T) {
	g := gomega.NewWithT(t)

	cpu := 0.0
	_, err := buildFunctionOptionsProto(&functionOptions{
		cpu: &cpu,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be a positive number"))
}

func TestBuildFunctionOptionsProto_NegativeMemory(t *testing.T) {
	g := gomega.NewWithT(t)

	memoryMiB := -100
	_, err := buildFunctionOptionsProto(&functionOptions{
		memoryMiB: &memoryMiB,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be a positive number"))
}

func TestBuildFunctionOptionsProto_ZeroMemory(t *testing.T) {
	g := gomega.NewWithT(t)

	memoryMiB := 0
	_, err := buildFunctionOptionsProto(&functionOptions{
		memoryMiB: &memoryMiB,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be a positive number"))
}

// mockClsRoutingClient is a minimal control-plane stub for a parameter-less
// Cls. It records the routing_region forwarded on FunctionBindParams and
// returns a bound variant whose per-method handle metadata carries input-plane
// routing fields.
type mockClsRoutingClient struct {
	pb.ModalClientClient
	gotRoutingRegion string
}

func (m *mockClsRoutingClient) FunctionGet(ctx context.Context, req *pb.FunctionGetRequest, opts ...grpc.CallOption) (*pb.FunctionGetResponse, error) {
	return pb.FunctionGetResponse_builder{
		FunctionId: "fid",
		HandleMetadata: pb.FunctionHandleMetadata_builder{
			MethodHandleMetadata: map[string]*pb.FunctionHandleMetadata{"echo_string": {}},
			ClassParameterInfo:   pb.ClassParameterInfo_builder{Schema: []*pb.ClassParameterSpec{}}.Build(),
		}.Build(),
	}.Build(), nil
}

func (m *mockClsRoutingClient) FunctionBindParams(ctx context.Context, req *pb.FunctionBindParamsRequest, opts ...grpc.CallOption) (*pb.FunctionBindParamsResponse, error) {
	m.gotRoutingRegion = req.GetFunctionOptions().GetRoutingRegion()
	inputPlaneURL := "https://us-east.modal.example"
	inputPlaneRegion := "us-east"
	return pb.FunctionBindParamsResponse_builder{
		BoundFunctionId: "fid-1",
		HandleMetadata: pb.FunctionHandleMetadata_builder{
			MethodHandleMetadata: map[string]*pb.FunctionHandleMetadata{
				"echo_string": pb.FunctionHandleMetadata_builder{
					InputPlaneUrl:    &inputPlaneURL,
					InputPlaneRegion: &inputPlaneRegion,
				}.Build(),
			},
		}.Build(),
	}.Build(), nil
}

// TestClsWithOptionsRoutingRegion verifies that RoutingRegion is forwarded to
// FunctionBindParams and that the instance's methods adopt the bound variant's
// per-method handle metadata (so input-plane routing takes effect at invocation
// time).
func TestClsWithOptionsRoutingRegion(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mockClient := &mockClsRoutingClient{}
	client, err := NewClientWithOptions(&ClientParams{
		TokenID:            "test-token-id",
		TokenSecret:        "test-token-secret",
		Environment:        "test",
		ControlPlaneClient: mockClient,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer client.Close()

	cls, err := client.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	region := "us-east"
	instance, err := cls.WithOptions(&ClsWithOptionsParams{RoutingRegion: &region}).Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	// routingRegion must be forwarded so the server creates a bound variant
	// with input-plane routing.
	g.Expect(mockClient.gotRoutingRegion).To(gomega.Equal("us-east"))

	// The instance's methods adopt the bound variant's per-method metadata.
	method, err := instance.Method("echo_string")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(method.FunctionID).To(gomega.Equal("fid-1"))
	g.Expect(method.handleMetadata.GetInputPlaneUrl()).To(gomega.Equal("https://us-east.modal.example"))
	g.Expect(method.handleMetadata.GetInputPlaneRegion()).To(gomega.Equal("us-east"))
}
