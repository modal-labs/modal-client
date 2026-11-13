package test

import (
	"errors"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	persistentSandboxID       = "sp-01ARZ3NDEKTSV4RRFFQ69G5FAV"
	persistentSessionID       = "sb-01ARZ3NDEKTSV4RRFFQ69G5FA1"
	persistentSecondSessionID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FA2"
)

// mockPersistentSandboxCreate serves the RPCs behind creating a persistent
// Sandbox and returns the created Sandbox. The create request is recorded
// into gotCreate when non-nil.
func mockPersistentSandboxCreate(t *testing.T, g *gomega.WithT, mock *grpcmock.MockClient, params *modal.SandboxCreateParams, gotCreate **pb.PersistentSandboxCreateRequest) *modal.Sandbox {
	t.Helper()
	grpcmock.HandleUnary(mock, "AppGetOrCreate",
		func(req *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
			return pb.AppGetOrCreateResponse_builder{AppId: "ap-1"}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "EnvironmentGetOrCreate",
		func(req *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
			return pb.EnvironmentGetOrCreateResponse_builder{
				EnvironmentId: "en-1",
				Metadata:      pb.EnvironmentMetadata_builder{Name: "libmodal", Settings: pb.EnvironmentSettings_builder{ImageBuilderVersion: "2025.06"}.Build()}.Build(),
			}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-1",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "PersistentSandboxCreate",
		func(req *pb.PersistentSandboxCreateRequest) (*pb.PersistentSandboxCreateResponse, error) {
			if gotCreate != nil {
				*gotCreate = req
			}
			return pb.PersistentSandboxCreateResponse_builder{
				PersistentSandboxId: persistentSandboxID,
				SandboxId:           persistentSessionID,
			}.Build(), nil
		})

	ctx := t.Context()
	app, err := mock.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	image := mock.Images.FromRegistry("alpine:3.21", nil)
	if params == nil {
		params = &modal.SandboxCreateParams{}
	}
	params.ExperimentalPersistent = true
	sb, err := mock.Sandboxes.Create(ctx, app, image, params)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	return sb
}

func handleConnectToken(mock *grpcmock.MockClient, g *gomega.WithT, wantSandboxID string, err error) {
	grpcmock.HandleUnary(mock, "SandboxCreateConnectTokenV2",
		func(req *pb.SandboxCreateConnectTokenRequest) (*pb.SandboxCreateConnectTokenResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(wantSandboxID))
			if err != nil {
				return nil, err
			}
			return pb.SandboxCreateConnectTokenResponse_builder{Url: "https://connect", Token: "tok"}.Build(), nil
		})
}

func TestPersistentSandboxCreateMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)

	var createReq *pb.PersistentSandboxCreateRequest
	sb := mockPersistentSandboxCreate(t, g, mock, &modal.SandboxCreateParams{Command: []string{"sleep", "infinity"}}, &createReq)

	g.Expect(sb.SandboxID).To(gomega.Equal(persistentSessionID))
	g.Expect(createReq.GetAppId()).To(gomega.Equal("ap-1"))
	def := createReq.GetDefinition()
	g.Expect(def.GetImageId()).To(gomega.Equal("im-1"))
	g.Expect(def.GetEntrypointArgs()).To(gomega.Equal([]string{"sleep", "infinity"}))
	g.Expect(def.GetTimeoutSecs()).To(gomega.Equal(uint32(60 * 60)))
	g.Expect(def.GetIdleTimeoutSecs()).To(gomega.Equal(uint32(5 * 60)))
	g.Expect(def.GetName()).To(gomega.BeEmpty())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxCreateRejectsUnsupportedOptionsMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()

	app := &modal.App{AppID: "ap-1"}
	image := mock.Images.FromRegistry("alpine:3.21", nil)
	for _, params := range []*modal.SandboxCreateParams{
		{Timeout: time.Minute},
		{IdleTimeout: time.Minute},
		{Name: "named"},
		{GPU: "A10G"},
		{PTY: true},
		{ExperimentalEnableSnapshot: true},
		{Env: map[string]string{"A": "b"}},
		{Tags: map[string]string{"k": "v"}},
	} {
		params.ExperimentalPersistent = true
		_, err := mock.Sandboxes.Create(ctx, app, image, params)
		var invalid modal.InvalidError
		g.Expect(errors.As(err, &invalid)).To(gomega.BeTrue(), "params %+v: %v", params, err)
	}
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxPauseAndResumeMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	grpcmock.HandleUnary(mock, "SandboxTerminateV2",
		func(req *pb.SandboxTerminateRequest) (*pb.SandboxTerminateResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(persistentSessionID))
			return pb.SandboxTerminateResponse_builder{}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxWaitV2",
		func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(persistentSessionID))
			return pb.SandboxWaitResponse_builder{
				Result: pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		})
	g.Expect(sb.ExperimentalPause(ctx, nil)).To(gomega.Succeed())

	// Pausing a session doesn't finish the persistent Sandbox.
	code, err := sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(code).To(gomega.BeNil())

	grpcmock.HandleUnary(mock, "PersistentSandboxResume",
		func(req *pb.PersistentSandboxResumeRequest) (*pb.PersistentSandboxResumeResponse, error) {
			g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
			return pb.PersistentSandboxResumeResponse_builder{SandboxId: persistentSecondSessionID}.Build(), nil
		})
	g.Expect(sb.ExperimentalResume(ctx, nil)).To(gomega.Succeed())

	// Operations now go to the new session; the handle's ID is unchanged.
	handleConnectToken(mock, g, persistentSecondSessionID, nil)
	creds, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(creds.Token).To(gomega.Equal("tok"))
	g.Expect(sb.SandboxID).To(gomega.Equal(persistentSessionID))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxTunnelsMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// Tunnels are listed for the persistent Sandbox, not its current session.
	grpcmock.HandleUnary(mock, "PersistentSandboxTunnelsList",
		func(req *pb.PersistentSandboxTunnelsListRequest) (*pb.PersistentSandboxTunnelsListResponse, error) {
			g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
			return pb.PersistentSandboxTunnelsListResponse_builder{
				Tunnels: []*pb.TunnelData{pb.TunnelData_builder{Host: "tunnel.modal.host", Port: 443, ContainerPort: 8080}.Build()},
			}.Build(), nil
		})
	tunnels, err := sb.Tunnels(ctx, time.Second, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(tunnels).To(gomega.HaveKeyWithValue(8080, gomega.HaveField("Host", "tunnel.modal.host")))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxRetriesOnReplacementSessionMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// The session stopped: the first attempt fails, the Sandbox is resumed
	// into a new session and the operation is retried there.
	handleConnectToken(mock, g, persistentSessionID, status.Error(codes.NotFound, "sandbox not found"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
			return pb.PersistentSandboxGetResponse_builder{PersistentSandboxId: persistentSandboxID}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "PersistentSandboxResume",
		func(req *pb.PersistentSandboxResumeRequest) (*pb.PersistentSandboxResumeResponse, error) {
			return nil, status.Error(codes.Unavailable, "still pausing")
		})
	grpcmock.HandleUnary(mock, "PersistentSandboxResume",
		func(req *pb.PersistentSandboxResumeRequest) (*pb.PersistentSandboxResumeResponse, error) {
			return pb.PersistentSandboxResumeResponse_builder{SandboxId: persistentSecondSessionID}.Build(), nil
		})
	handleConnectToken(mock, g, persistentSecondSessionID, nil)

	creds, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(creds.Token).To(gomega.Equal("tok"))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxSwitchesToRunningSessionWithoutResumeMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	handleConnectToken(mock, g, persistentSessionID, status.Error(codes.NotFound, "sandbox not found"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			return pb.PersistentSandboxGetResponse_builder{
				PersistentSandboxId: persistentSandboxID,
				SandboxId:           persistentSecondSessionID,
			}.Build(), nil
		})
	handleConnectToken(mock, g, persistentSecondSessionID, nil)

	_, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxDoesNotRetryOnRunningSessionMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// The failure wasn't caused by the session having stopped, so it surfaces.
	handleConnectToken(mock, g, persistentSessionID, status.Error(codes.Internal, "boom"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			return pb.PersistentSandboxGetResponse_builder{
				PersistentSandboxId: persistentSandboxID,
				SandboxId:           persistentSessionID,
			}.Build(), nil
		})

	_, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{})
	g.Expect(status.Code(err)).To(gomega.Equal(codes.Internal))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxTerminateMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	code, err := sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(code).To(gomega.BeNil())

	grpcmock.HandleUnary(mock, "PersistentSandboxTerminate",
		func(req *pb.PersistentSandboxTerminateRequest) (*pb.PersistentSandboxTerminateResponse, error) {
			g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
			return pb.PersistentSandboxTerminateResponse_builder{}.Build(), nil
		})
	exitCode, err := sb.Terminate(ctx, &modal.SandboxTerminateParams{Wait: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(137))

	code, err = sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(code).To(gomega.HaveValue(gomega.Equal(137)))

	// A terminated persistent Sandbox is not resumed on use.
	handleConnectToken(mock, g, persistentSessionID, status.Error(codes.NotFound, "sandbox not found"))
	_, err = sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{})
	g.Expect(status.Code(err)).To(gomega.Equal(codes.NotFound))

	var invalid modal.InvalidError
	g.Expect(errors.As(sb.ExperimentalResume(ctx, nil), &invalid)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxUnsupportedOperationsMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	var invalid modal.InvalidError
	g.Expect(errors.As(sb.SetTags(ctx, map[string]string{"k": "v"}, nil), &invalid)).To(gomega.BeTrue())
	_, err := sb.GetTags(ctx, nil)
	g.Expect(errors.As(err, &invalid)).To(gomega.BeTrue())
	g.Expect(errors.As(sb.ExperimentalSetName(ctx, "name", nil), &invalid)).To(gomega.BeTrue())
	_, err = sb.ExperimentalSnapshot(ctx, nil)
	g.Expect(errors.As(err, &invalid)).To(gomega.BeTrue())
	g.Expect(errors.As(sb.MountImage(ctx, "/mnt", nil, nil), &invalid)).To(gomega.BeTrue())
	g.Expect(errors.As(sb.UnmountImage(ctx, "/mnt", nil), &invalid)).To(gomega.BeTrue())
	g.Expect(errors.As(sb.UpdateNetworkPolicy(ctx, &modal.SandboxUpdateNetworkPolicyParams{
		OutboundCIDRAllowlist:   &modal.Allowlist{},
		OutboundDomainAllowlist: &modal.Allowlist{},
	}), &invalid)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPauseResumeRequirePersistentSandboxMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()

	for _, id := range []string{"sb-nGEijt9WbBMlGrsPH9FOaC", "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"} {
		sb, err := mock.Sandboxes.FromID(ctx, id, nil)
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		var invalid modal.InvalidError
		g.Expect(errors.As(sb.ExperimentalPause(ctx, nil), &invalid)).To(gomega.BeTrue())
		g.Expect(errors.As(sb.ExperimentalResume(ctx, nil), &invalid)).To(gomega.BeTrue())
	}
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
