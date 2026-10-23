package test

import (
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestAppStop(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "/AppGetOrCreate",
		func(req *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
			return pb.AppGetOrCreateResponse_builder{AppId: "ap-test-123"}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "/AppStop",
		func(req *pb.AppStopRequest) (*emptypb.Empty, error) {
			g.Expect(req.GetAppId()).To(gomega.Equal("ap-test-123"))
			g.Expect(req.GetSource()).To(gomega.Equal(pb.AppStopSource_APP_STOP_SOURCE_PYTHON_CLIENT))
			return &emptypb.Empty{}, nil
		},
	)

	app, err := mock.Apps.FromName(ctx, "libmodal-test", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = app.Stop(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestAppStopWithoutClient(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	app := &modal.App{AppID: "ap-test-123"}
	err := app.Stop(t.Context(), nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
}

func TestAppStopWithoutAppID(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	app := &modal.App{}
	err := app.Stop(t.Context(), nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
	g.Expect(err.Error()).To(gomega.ContainSubstring("without an ID"))
}
