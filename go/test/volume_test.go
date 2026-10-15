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

func TestVolumeFromNameVersion(t *testing.T) {
	tests := []struct {
		name                    string
		params                  *modal.VolumeFromNameParams
		responseVersion         pb.VolumeFsVersion
		wantVersion             pb.VolumeFsVersion
		wantCreationType        pb.ObjectCreationType
		wantError               string
		expectVolumeGetOrCreate bool
	}{
		{
			name:                    "v2 with create if missing",
			params:                  &modal.VolumeFromNameParams{Version: 2, CreateIfMissing: true},
			responseVersion:         pb.VolumeFsVersion_VOLUME_FS_VERSION_V2,
			wantVersion:             pb.VolumeFsVersion_VOLUME_FS_VERSION_V2,
			wantCreationType:        pb.ObjectCreationType_OBJECT_CREATION_TYPE_CREATE_IF_MISSING,
			expectVolumeGetOrCreate: true,
		},
		{
			name:                    "v2 rejects v1",
			params:                  &modal.VolumeFromNameParams{Version: 2},
			responseVersion:         pb.VolumeFsVersion_VOLUME_FS_VERSION_V1,
			wantVersion:             pb.VolumeFsVersion_VOLUME_FS_VERSION_V2,
			wantError:               "Volume 'test-volume' exists but has version v1, not v2 as requested. To access this Volume, either omit the version parameter or use the correct version.",
			expectVolumeGetOrCreate: true,
		},
		{
			name:                    "v1 accepts unspecified response",
			params:                  &modal.VolumeFromNameParams{Version: 1},
			responseVersion:         pb.VolumeFsVersion_VOLUME_FS_VERSION_UNSPECIFIED,
			wantVersion:             pb.VolumeFsVersion_VOLUME_FS_VERSION_V1,
			expectVolumeGetOrCreate: true,
		},
		{
			name:                    "v2 accepts v2 response",
			params:                  &modal.VolumeFromNameParams{Version: 2},
			responseVersion:         pb.VolumeFsVersion_VOLUME_FS_VERSION_V2,
			wantVersion:             pb.VolumeFsVersion_VOLUME_FS_VERSION_V2,
			expectVolumeGetOrCreate: true,
		},
		{
			name:      "invalid version does not make an RPC",
			params:    &modal.VolumeFromNameParams{Version: 3},
			wantError: "VolumeFS version must be either 1 or 2",
		},
		{
			name:                    "omitted version accepts any response",
			responseVersion:         pb.VolumeFsVersion_VOLUME_FS_VERSION_V2,
			wantVersion:             pb.VolumeFsVersion_VOLUME_FS_VERSION_UNSPECIFIED,
			expectVolumeGetOrCreate: true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			mock := newGRPCMockClient(t)

			if tt.expectVolumeGetOrCreate {
				grpcmock.HandleUnary(mock, "/VolumeGetOrCreate",
					func(req *pb.VolumeGetOrCreateRequest) (*pb.VolumeGetOrCreateResponse, error) {
						g.Expect(req.GetVersion()).To(gomega.Equal(tt.wantVersion))
						g.Expect(req.GetObjectCreationType()).To(gomega.Equal(tt.wantCreationType))
						return pb.VolumeGetOrCreateResponse_builder{
							VolumeId: "vo-test",
							Metadata: pb.VolumeMetadata_builder{Version: tt.responseVersion}.Build(),
						}.Build(), nil
					},
				)
			}

			volume, err := mock.Volumes.FromName(t.Context(), "test-volume", tt.params)
			if tt.wantError != "" {
				g.Expect(err).Should(gomega.HaveOccurred())
				g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
				g.Expect(err).Should(gomega.MatchError("InvalidError: " + tt.wantError))
			} else {
				g.Expect(err).ShouldNot(gomega.HaveOccurred())
				g.Expect(volume).ShouldNot(gomega.BeNil())
			}
			g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
		})
	}
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
