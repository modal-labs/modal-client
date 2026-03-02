package test

import (
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestVolumeFromName(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	volume, err := tc.Volumes.FromName(ctx, "libmodal-test-volume", &modal.VolumeFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(volume).ShouldNot(gomega.BeNil())
	g.Expect(volume.VolumeID).Should(gomega.HavePrefix("vo-"))
	g.Expect(volume.Name).To(gomega.Equal("libmodal-test-volume"))

	_, err = tc.Volumes.FromName(ctx, "missing-volume", nil)
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("Volume 'missing-volume' not found")))
}

func TestVolumeReadOnly(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	volume, err := tc.Volumes.FromName(ctx, "libmodal-test-volume", &modal.VolumeFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(volume.IsReadOnly()).To(gomega.BeFalse())

	readOnlyVolume := volume.ReadOnly()
	g.Expect(readOnlyVolume.IsReadOnly()).To(gomega.BeTrue())
	g.Expect(readOnlyVolume.VolumeID).To(gomega.Equal(volume.VolumeID))
	g.Expect(readOnlyVolume.Name).To(gomega.Equal(volume.Name))

	g.Expect(volume.IsReadOnly()).To(gomega.BeFalse())
}

func TestVolumeEphemeral(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	tc := newTestClient(t)

	volume, err := tc.Volumes.Ephemeral(t.Context(), nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer volume.CloseEphemeral()
	g.Expect(volume.Name).To(gomega.BeEmpty())
	g.Expect(volume.VolumeID).Should(gomega.HavePrefix("vo-"))
	g.Expect(volume.IsReadOnly()).To(gomega.BeFalse())
	g.Expect(volume.ReadOnly().IsReadOnly()).To(gomega.BeTrue())
}

func TestVolumeDeleteSuccess(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/VolumeGetOrCreate",
		func(req *pb.VolumeGetOrCreateRequest) (*pb.VolumeGetOrCreateResponse, error) {
			return pb.VolumeGetOrCreateResponse_builder{
				VolumeId: "vo-test-123",
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "/VolumeDelete",
		func(req *pb.VolumeDeleteRequest) (*emptypb.Empty, error) {
			g.Expect(req.GetVolumeId()).To(gomega.Equal("vo-test-123"))
			return &emptypb.Empty{}, nil
		},
	)

	err := mock.Volumes.Delete(ctx, "test-volume", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestVolumeDeleteWithAllowMissing(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/VolumeGetOrCreate",
		func(req *pb.VolumeGetOrCreateRequest) (*pb.VolumeGetOrCreateResponse, error) {
			return nil, modal.NotFoundError{Exception: "Volume 'missing' not found"}
		},
	)

	err := mock.Volumes.Delete(ctx, "missing", &modal.VolumeDeleteParams{
		AllowMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestVolumeDeleteWithAllowMissingDeleteRPCNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "/VolumeGetOrCreate",
		func(req *pb.VolumeGetOrCreateRequest) (*pb.VolumeGetOrCreateResponse, error) {
			return pb.VolumeGetOrCreateResponse_builder{VolumeId: "vo-test-123"}.Build(), nil
		},
	)

	grpcmock.HandleUnary(mock, "/VolumeDelete",
		func(req *pb.VolumeDeleteRequest) (*emptypb.Empty, error) {
			return nil, status.Errorf(codes.NotFound, "Volume not found")
		},
	)

	err := mock.Volumes.Delete(ctx, "test-volume", &modal.VolumeDeleteParams{AllowMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestVolumeDeleteWithAllowMissingFalseThrows(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/VolumeGetOrCreate",
		func(req *pb.VolumeGetOrCreateRequest) (*pb.VolumeGetOrCreateResponse, error) {
			return nil, modal.NotFoundError{Exception: "Volume 'missing' not found"}
		},
	)

	err := mock.Volumes.Delete(ctx, "missing", &modal.VolumeDeleteParams{
		AllowMissing: false,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	var notFoundErr modal.NotFoundError
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(notFoundErr))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
