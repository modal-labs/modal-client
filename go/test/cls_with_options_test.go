package test

import (
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

var mockFunctionProto = pb.FunctionGetResponse_builder{
	FunctionId: "fid",
	HandleMetadata: pb.FunctionHandleMetadata_builder{
		MethodHandleMetadata: map[string]*pb.FunctionHandleMetadata{"echo_string": {}},
		ClassParameterInfo:   pb.ClassParameterInfo_builder{Schema: []*pb.ClassParameterSpec{}}.Build(),
	}.Build(),
}.Build()

func TestClsWithOptionsStacking(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return mockFunctionProto, nil
		},
	)

	cls, err := mock.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "FunctionBindParams",
		func(req *pb.FunctionBindParamsRequest) (*pb.FunctionBindParamsResponse, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid"))
			fo := req.GetFunctionOptions()
			g.Expect(fo).ToNot(gomega.BeNil())
			g.Expect(fo.GetTimeoutSecs()).To(gomega.Equal(uint32(60)))
			g.Expect(fo.GetResources()).ToNot(gomega.BeNil())
			g.Expect(fo.GetResources().GetMilliCpu()).To(gomega.Equal(uint32(250)))
			g.Expect(fo.GetResources().GetMemoryMb()).To(gomega.Equal(uint32(256)))
			g.Expect(fo.GetResources().GetGpuConfig()).ToNot(gomega.BeNil())
			g.Expect(fo.GetSecretIds()).To(gomega.Equal([]string{"sec-1"}))
			g.Expect(fo.GetReplaceSecretIds()).To(gomega.BeTrue())
			g.Expect(fo.GetReplaceVolumeMounts()).To(gomega.BeTrue())
			g.Expect(fo.GetVolumeMounts()).To(gomega.HaveLen(1))
			g.Expect(fo.GetVolumeMounts()[0].GetMountPath()).To(gomega.Equal("/mnt/test"))
			g.Expect(fo.GetVolumeMounts()[0].GetVolumeId()).To(gomega.Equal("vol-1"))
			g.Expect(fo.GetVolumeMounts()[0].GetAllowBackgroundCommits()).To(gomega.BeTrue())
			g.Expect(fo.GetVolumeMounts()[0].GetReadOnly()).To(gomega.BeFalse())
			return pb.FunctionBindParamsResponse_builder{BoundFunctionId: "fid-1", HandleMetadata: &pb.FunctionHandleMetadata{}}.Build(), nil
		},
	)

	secret := &modal.Secret{SecretID: "sec-1"}
	volume := &modal.Volume{VolumeID: "vol-1"}
	cpu := 0.25
	memoryMiB := 256
	gpu := "T4"
	timeout := 45 * time.Second
	newTimeout := 60 * time.Second

	optioned := cls.
		WithOptions(&modal.ClsWithOptionsParams{Timeout: &timeout, CPU: &cpu}).
		WithOptions(&modal.ClsWithOptionsParams{Timeout: &newTimeout, MemoryMiB: &memoryMiB, GPU: &gpu}).
		WithOptions(&modal.ClsWithOptionsParams{Secrets: []*modal.Secret{secret}, Volumes: map[string]*modal.Volume{"/mnt/test": volume}})

	instance, err := optioned.Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(instance).ToNot(gomega.BeNil())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestClsWithConcurrencyWithBatchingChaining(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return mockFunctionProto, nil
		},
	)

	cls, err := mock.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "FunctionBindParams",
		func(req *pb.FunctionBindParamsRequest) (*pb.FunctionBindParamsResponse, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid"))
			fo := req.GetFunctionOptions()
			g.Expect(fo).ToNot(gomega.BeNil())
			g.Expect(fo.GetTimeoutSecs()).To(gomega.Equal(uint32(60)))
			g.Expect(fo.GetMaxConcurrentInputs()).To(gomega.Equal(uint32(10)))
			g.Expect(fo.GetBatchMaxSize()).To(gomega.Equal(uint32(11)))
			g.Expect(fo.GetBatchLingerMs()).To(gomega.Equal(uint64(12)))
			return pb.FunctionBindParamsResponse_builder{BoundFunctionId: "fid-1", HandleMetadata: &pb.FunctionHandleMetadata{}}.Build(), nil
		},
	)

	timeout := 60 * time.Second
	chained := cls.
		WithOptions(&modal.ClsWithOptionsParams{Timeout: &timeout}).
		WithConcurrency(&modal.ClsWithConcurrencyParams{MaxInputs: 10}).
		WithBatching(&modal.ClsWithBatchingParams{MaxBatchSize: 11, Wait: 12 * time.Millisecond})

	instance, err := chained.Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(instance).ToNot(gomega.BeNil())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestClsWithOptionsRetries(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return mockFunctionProto, nil
		},
	)

	cls, err := mock.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "FunctionBindParams",
		func(req *pb.FunctionBindParamsRequest) (*pb.FunctionBindParamsResponse, error) {
			fo := req.GetFunctionOptions()
			g.Expect(fo).ToNot(gomega.BeNil())
			g.Expect(fo.GetRetryPolicy()).ToNot(gomega.BeNil())
			g.Expect(fo.GetRetryPolicy().GetRetries()).To(gomega.Equal(uint32(2)))
			g.Expect(fo.GetRetryPolicy().GetBackoffCoefficient()).To(gomega.Equal(float32(2.0)))
			g.Expect(fo.GetRetryPolicy().GetInitialDelayMs()).To(gomega.Equal(uint32(2000)))
			g.Expect(fo.GetRetryPolicy().GetMaxDelayMs()).To(gomega.Equal(uint32(5000)))
			return pb.FunctionBindParamsResponse_builder{BoundFunctionId: "fid-2", HandleMetadata: &pb.FunctionHandleMetadata{}}.Build(), nil
		},
	)

	backoff := float32(2.0)
	initial := 2 * time.Second
	max := 5 * time.Second
	retries, err := modal.NewRetries(2, &modal.RetriesParams{
		BackoffCoefficient: &backoff,
		InitialDelay:       &initial,
		MaxDelay:           &max,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{Retries: retries}).Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestClsWithOptionsInvalidValues(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return mockFunctionProto, nil
		},
	)

	cls, err := mock.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	timeout := 500 * time.Millisecond
	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{Timeout: &timeout}).Instance(ctx, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("timeout must be at least 1 second"))

	scaledownWindow := 100 * time.Millisecond
	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{ScaledownWindow: &scaledownWindow}).Instance(ctx, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("scaledownWindow must be at least 1 second"))

	fractionalTimeout := 1500 * time.Millisecond
	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{Timeout: &fractionalTimeout}).Instance(ctx, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("whole number of seconds"))

	fractionalScaledown := 1500 * time.Millisecond
	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{ScaledownWindow: &fractionalScaledown}).Instance(ctx, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("whole number of seconds"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestWithOptionsEmptySecretsDoesNotReplace(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return mockFunctionProto, nil
		},
	)

	cls, err := mock.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "FunctionBindParams",
		func(req *pb.FunctionBindParamsRequest) (*pb.FunctionBindParamsResponse, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid"))
			fo := req.GetFunctionOptions()
			g.Expect(fo.GetSecretIds()).To(gomega.HaveLen(0))
			g.Expect(fo.GetReplaceSecretIds()).To(gomega.BeFalse())

			return pb.FunctionBindParamsResponse_builder{BoundFunctionId: "fid-1", HandleMetadata: &pb.FunctionHandleMetadata{}}.Build(), nil
		},
	)

	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{Secrets: []*modal.Secret{}}).Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestWithOptionsEmptyVolumesDoesNotReplace(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "FunctionGet",
		func(req *pb.FunctionGetRequest) (*pb.FunctionGetResponse, error) {
			return mockFunctionProto, nil
		},
	)

	cls, err := mock.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	grpcmock.HandleUnary(
		mock, "FunctionBindParams",
		func(req *pb.FunctionBindParamsRequest) (*pb.FunctionBindParamsResponse, error) {
			g.Expect(req.GetFunctionId()).To(gomega.Equal("fid"))
			fo := req.GetFunctionOptions()
			g.Expect(fo.GetVolumeMounts()).To(gomega.HaveLen(0))
			g.Expect(fo.GetReplaceVolumeMounts()).To(gomega.BeFalse())

			return pb.FunctionBindParamsResponse_builder{BoundFunctionId: "fid-1", HandleMetadata: &pb.FunctionHandleMetadata{}}.Build(), nil
		},
	)

	_, err = cls.WithOptions(&modal.ClsWithOptionsParams{Volumes: map[string]*modal.Volume{}}).Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
