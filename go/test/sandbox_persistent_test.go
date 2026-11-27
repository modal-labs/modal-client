package test

import (
	"context"
	"errors"
	"io"
	"os"
	"sync/atomic"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const (
	persistentSandboxID       = "sx-z1arz3ndektsv4rrffq69g5fa"
	persistentSessionID       = "sb-01ARZ3NDEKTSV4RRFFQ69G5FA1"
	persistentSecondSessionID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FA2"
)

func TestPersistentSandboxFromIDMock(t *testing.T) {
	for _, state := range []string{"running", "paused", "terminated", "missing", "stale"} {
		t.Run(state, func(t *testing.T) {
			g := gomega.NewWithT(t)
			mock := newGRPCMockClient(t)
			ctx := t.Context()
			var calls atomic.Int32
			get := func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
				calls.Add(1)
				g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
				if state == "missing" {
					return nil, status.Error(codes.NotFound, "unknown")
				}
				resp := pb.PersistentSandboxGetResponse_builder{PersistentSandboxId: persistentSandboxID}
				if state == "running" || state == "stale" {
					resp.CurrentSandboxId = proto.String(persistentSessionID)
				}
				if state == "terminated" {
					resp.TerminatedAt = timestamppb.New(time.Unix(0, 0))
				}
				return resp.Build(), nil
			}
			grpcmock.HandleUnary(mock, "PersistentSandboxGet", get)
			sb, err := mock.Sandboxes.FromID(ctx, persistentSandboxID, nil)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(calls.Load()).To(gomega.BeZero())
			result, err := sb.Poll(ctx, nil)
			g.Expect(calls.Load()).To(gomega.Equal(int32(1)))
			if state == "missing" {
				g.Expect(err).To(gomega.HaveOccurred())
				g.Expect(status.Code(err)).To(gomega.Equal(codes.NotFound))
			} else {
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(sb.SandboxID).To(gomega.Equal(persistentSandboxID))
				g.Expect(err).NotTo(gomega.HaveOccurred())
				if state == "terminated" {
					g.Expect(result).NotTo(gomega.BeNil())
					_, err = sb.Wait(ctx, nil)
					g.Expect(err).NotTo(gomega.HaveOccurred())
					_, err = createSidecar(t, sb)
					g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("has been terminated")))
					g.Expect(sb.ExperimentalPause(ctx, nil)).To(gomega.HaveOccurred())
					g.Expect(sb.ExperimentalResume(ctx, nil)).To(gomega.HaveOccurred())
				} else {
					g.Expect(result).To(gomega.BeNil())
					if state == "paused" {
						g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
						grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
							return pb.PersistentSandboxGetResponse_builder{}.Build(), nil
						})
						g.Expect(sb.ExperimentalPause(ctx, nil)).To(gomega.Succeed())
					}
					sessionID := persistentSessionID
					if state != "running" {
						if state == "stale" {
							handleSidecarCreate(mock, g, persistentSessionID, status.Error(codes.NotFound, "task gone"))
						}
						grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
							return pb.PersistentSandboxGetResponse_builder{}.Build(), nil
						})
						_, err := createSidecar(t, sb)
						var paused modal.SandboxPaused
						g.Expect(errors.As(err, &paused)).To(gomega.BeTrue())
						g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
						grpcmock.HandleUnary(mock, "PersistentSandboxResume", func(req *pb.PersistentSandboxResumeRequest) (*pb.PersistentSandboxResumeResponse, error) {
							g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
							return pb.PersistentSandboxResumeResponse_builder{SandboxId: persistentSecondSessionID}.Build(), nil
						})
						g.Expect(sb.ExperimentalResume(ctx, nil)).To(gomega.Succeed())
						sessionID = persistentSecondSessionID
					}
					handleSidecarCreate(mock, g, sessionID, nil)
					sidecarID, err := createSidecar(t, sb)
					g.Expect(err).NotTo(gomega.HaveOccurred())
					g.Expect(sidecarID).To(gomega.Equal(persistentSidecarID))
				}
			}
			g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
		})
	}
}

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

const persistentSidecarID = "sb-ctr-1"

// handleSidecarCreate serves one sidecar create. Sidecar create is a single RPC
// on the current session, which makes it a convenient probe for the operations
// that follow the persistent Sandbox to its running session. Tests using it set
// the sidecar create opt-out env var, so they can't run in parallel.
func handleSidecarCreate(mock *grpcmock.MockClient, g *gomega.WithT, wantSandboxID string, err error) {
	grpcmock.HandleUnary(mock, "SandboxContainerCreateV2",
		func(req *pb.SandboxContainerCreateV2Request) (*pb.SandboxContainerCreateV2Response, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(wantSandboxID))
			if err != nil {
				return nil, err
			}
			return pb.SandboxContainerCreateV2Response_builder{ContainerId: persistentSidecarID}.Build(), nil
		})
}

func createSidecar(t *testing.T, sb *modal.Sandbox) (string, error) {
	t.Helper()
	// An empty value still opts out, so the variable is removed after t.Setenv
	// arranges for it to be restored.
	t.Setenv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", "")
	if err := os.Unsetenv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE"); err != nil {
		t.Fatal(err)
	}
	sidecar, err := sb.ExperimentalSidecars.Create(t.Context(), "worker", &modal.Image{ImageID: "im-1"}, nil)
	if err != nil {
		return "", err
	}
	return sidecar.ContainerID, nil
}

func TestPersistentSandboxCreateMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)

	var createReq *pb.PersistentSandboxCreateRequest
	sb := mockPersistentSandboxCreate(t, g, mock, &modal.SandboxCreateParams{Command: []string{"sleep", "infinity"}}, &createReq)

	g.Expect(sb.SandboxID).To(gomega.Equal(persistentSandboxID))
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
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// Pausing looks up the running session, and stops that one.
	handlePersistentGet(mock, persistentSessionID)
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
	grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
		return pb.PersistentSandboxGetResponse_builder{}.Build(), nil
	})
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
	handleSidecarCreate(mock, g, persistentSecondSessionID, nil)
	sidecarID, err := createSidecar(t, sb)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sidecarID).To(gomega.Equal(persistentSidecarID))
	g.Expect(sb.SandboxID).To(gomega.Equal(persistentSandboxID))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

// handlePersistentGet answers one lookup with the given running session, or a
// paused persistent Sandbox if it is empty.
func handlePersistentGet(mock *grpcmock.MockClient, sessionID string) {
	grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
		return pb.PersistentSandboxGetResponse_builder{PersistentSandboxId: persistentSandboxID, CurrentSandboxId: proto.String(sessionID)}.Build(), nil
	})
}

func TestPersistentSandboxPauseStopsSessionStartedElsewhereMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// Another handle paused and resumed the Sandbox, so a different session is running.
	handlePersistentGet(mock, persistentSecondSessionID)
	grpcmock.HandleUnary(mock, "SandboxTerminateV2",
		func(req *pb.SandboxTerminateRequest) (*pb.SandboxTerminateResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(persistentSecondSessionID))
			return pb.SandboxTerminateResponse_builder{}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxWaitV2",
		func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(persistentSecondSessionID))
			return pb.SandboxWaitResponse_builder{
				Result: pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		})
	g.Expect(sb.ExperimentalPause(t.Context(), nil)).To(gomega.Succeed())
}

func TestPersistentSandboxPauseDoesNotResumeMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// With no running session there is nothing to pause, and no other request is made.
	handlePersistentGet(mock, "")
	g.Expect(sb.ExperimentalPause(t.Context(), nil)).To(gomega.Succeed())
}

func TestPersistentSandboxExitSnapshotRequiresTerminationMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// A paused Sandbox has not exited: it is neither resumed nor asked for a snapshot.
	handlePersistentGet(mock, "")
	timeout := time.Duration(0)
	_, err := sb.ExperimentalGetExitSnapshot(t.Context(), &modal.SandboxExperimentalGetExitSnapshotParams{Timeout: &timeout})
	g.Expect(err).To(gomega.BeAssignableToTypeOf(modal.TimeoutError{}))
}

func TestPersistentSandboxExitSnapshotAfterTerminateMock(t *testing.T) {
	t.Parallel()
	for _, terminatedBy := range []string{"this handle", "another handle"} {
		t.Run(terminatedBy, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			mock := newGRPCMockClient(t)
			ctx := t.Context()
			sb, err := mock.Sandboxes.FromID(ctx, persistentSandboxID, nil)
			g.Expect(err).ToNot(gomega.HaveOccurred())

			if terminatedBy == "this handle" {
				// Terminating reports the last session, so no lookup is needed.
				grpcmock.HandleUnary(mock, "PersistentSandboxTerminate",
					func(*pb.PersistentSandboxTerminateRequest) (*pb.PersistentSandboxTerminateResponse, error) {
						return pb.PersistentSandboxTerminateResponse_builder{LastSessionSandboxId: proto.String(persistentSecondSessionID)}.Build(), nil
					})
				_, err = sb.Terminate(ctx, nil)
				g.Expect(err).ToNot(gomega.HaveOccurred())
			} else {
				grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
					return pb.PersistentSandboxGetResponse_builder{
						CurrentSandboxId: proto.String(persistentSecondSessionID),
						TerminatedAt:     timestamppb.Now(),
					}.Build(), nil
				})
			}
			grpcmock.HandleUnary(mock, "SandboxGetExitSnapshotV2",
				func(req *pb.SandboxGetExitSnapshotRequest) (*pb.SandboxGetExitSnapshotResponse, error) {
					g.Expect(req.GetSandboxId()).To(gomega.Equal(persistentSecondSessionID))
					return pb.SandboxGetExitSnapshotResponse_builder{
						Success: pb.SandboxGetExitSnapshotResponse_Success_builder{ImageId: "im-exit-snapshot-123"}.Build(),
					}.Build(), nil
				})
			timeout := time.Duration(0)
			image, err := sb.ExperimentalGetExitSnapshot(ctx, &modal.SandboxExperimentalGetExitSnapshotParams{Timeout: &timeout})
			g.Expect(err).ToNot(gomega.HaveOccurred())
			g.Expect(image.ImageID).To(gomega.Equal("im-exit-snapshot-123"))
			g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
		})
	}
}

func TestPersistentSandboxLogsDoNotResumeMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb, err := mock.Sandboxes.FromID(t.Context(), persistentSandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	// A paused Sandbox with no known session has no logs to read, and is not resumed for them.
	handlePersistentGet(mock, "")
	_, err = sb.Logs.Fetch(t.Context(), time.Time{}, nil)
	g.Expect(err).To(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
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

func TestPersistentSandboxCreateConnectTokenMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// Connect tokens are created for the persistent Sandbox, not its current session.
	grpcmock.HandleUnary(mock, "PersistentSandboxCreateConnectToken",
		func(req *pb.PersistentSandboxCreateConnectTokenRequest) (*pb.PersistentSandboxCreateConnectTokenResponse, error) {
			g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
			g.Expect(req.GetPort()).To(gomega.Equal(uint32(3000)))
			g.Expect(req.GetUserMetadata()).To(gomega.Equal("alice"))
			return pb.PersistentSandboxCreateConnectTokenResponse_builder{Url: "https://a-1-abcde.modal.host", Token: "tok"}.Build(), nil
		})
	creds, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{Port: 3000, UserMetadata: "alice"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(creds.URL).To(gomega.Equal("https://a-1-abcde.modal.host"))
	g.Expect(creds.Token).To(gomega.Equal("tok"))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxRequiresExplicitResumeMock(t *testing.T) {
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	handleSidecarCreate(mock, g, persistentSessionID, status.Error(codes.NotFound, "sandbox not found"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			g.Expect(req.GetPersistentSandboxId()).To(gomega.Equal(persistentSandboxID))
			return pb.PersistentSandboxGetResponse_builder{
				PersistentSandboxId: persistentSandboxID,
				CurrentSandboxId:    proto.String(persistentSessionID),
			}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxWaitV2",
		func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(persistentSessionID))
			return pb.SandboxWaitResponse_builder{
				Result: pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_TERMINATED}.Build(),
			}.Build(), nil
		})
	_, err := createSidecar(t, sb)
	var paused modal.SandboxPaused
	g.Expect(errors.As(err, &paused)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).To(gomega.Succeed())

	grpcmock.HandleUnary(mock, "PersistentSandboxResume",
		func(req *pb.PersistentSandboxResumeRequest) (*pb.PersistentSandboxResumeResponse, error) {
			return nil, status.Error(codes.Unavailable, "still pausing")
		})
	grpcmock.HandleUnary(mock, "PersistentSandboxResume",
		func(req *pb.PersistentSandboxResumeRequest) (*pb.PersistentSandboxResumeResponse, error) {
			return pb.PersistentSandboxResumeResponse_builder{SandboxId: persistentSecondSessionID}.Build(), nil
		})
	handleSidecarCreate(mock, g, persistentSecondSessionID, nil)

	g.Expect(sb.ExperimentalResume(t.Context(), nil)).To(gomega.Succeed())
	sidecarID, err := createSidecar(t, sb)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sidecarID).To(gomega.Equal(persistentSidecarID))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxStaleHandleRequiresExplicitResumeMock(t *testing.T) {
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	handleSidecarCreate(mock, g, persistentSessionID, status.Error(codes.NotFound, "sandbox not found"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			return pb.PersistentSandboxGetResponse_builder{
				PersistentSandboxId: persistentSandboxID,
				CurrentSandboxId:    proto.String(persistentSecondSessionID),
			}.Build(), nil
		})
	_, err := createSidecar(t, sb)
	var paused modal.SandboxPaused
	g.Expect(errors.As(err, &paused)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxDoesNotRetryOnRunningSessionMock(t *testing.T) {
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	// The failure wasn't caused by the session having stopped, so it surfaces.
	handleSidecarCreate(mock, g, persistentSessionID, status.Error(codes.Internal, "boom"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			return pb.PersistentSandboxGetResponse_builder{
				PersistentSandboxId: persistentSandboxID,
				CurrentSandboxId:    proto.String(persistentSessionID),
			}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxWaitV2",
		func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
			return pb.SandboxWaitResponse_builder{}.Build(), nil
		})

	_, err := createSidecar(t, sb)
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("boom")))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxTerminateMock(t *testing.T) {
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
		return pb.PersistentSandboxGetResponse_builder{CurrentSandboxId: proto.String(persistentSessionID)}.Build(), nil
	})
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

	// Operations fail without a request to the last session.
	var invalid modal.InvalidError
	_, err = createSidecar(t, sb)
	g.Expect(errors.As(err, &invalid)).To(gomega.BeTrue())

	g.Expect(errors.As(sb.ExperimentalResume(ctx, nil), &invalid)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxExternalTerminationMock(t *testing.T) {
	for _, sessionID := range []string{persistentSessionID, ""} {
		for _, operation := range []string{"poll", "wait"} {
			t.Run(sessionID+"/"+operation, func(t *testing.T) {
				t.Parallel()
				g := gomega.NewWithT(t)
				mock := newGRPCMockClient(t)
				ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
				defer cancel()
				get := func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
					return pb.PersistentSandboxGetResponse_builder{CurrentSandboxId: proto.String(sessionID)}.Build(), nil
				}
				sandboxA, err := mock.Sandboxes.FromID(ctx, persistentSandboxID, nil)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				sandboxB, err := mock.Sandboxes.FromID(ctx, persistentSandboxID, nil)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				grpcmock.HandleUnary(mock, "PersistentSandboxTerminate", func(*pb.PersistentSandboxTerminateRequest) (*pb.PersistentSandboxTerminateResponse, error) {
					return pb.PersistentSandboxTerminateResponse_builder{}.Build(), nil
				})
				if operation == "poll" {
					grpcmock.HandleUnary(mock, "PersistentSandboxGet", get)
					code, err := sandboxB.Poll(ctx, nil)
					g.Expect(err).NotTo(gomega.HaveOccurred())
					g.Expect(code).To(gomega.BeNil())
					_, err = sandboxA.Terminate(ctx, nil)
					g.Expect(err).NotTo(gomega.HaveOccurred())
				} else {
					grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
						_, err := sandboxA.Terminate(ctx, nil)
						g.Expect(err).NotTo(gomega.HaveOccurred())
						return get(req)
					})
				}
				grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
					return pb.PersistentSandboxGetResponse_builder{TerminatedAt: timestamppb.New(time.Unix(0, 0))}.Build(), nil
				})
				if operation == "wait" {
					code, err := sandboxB.Wait(ctx, nil)
					g.Expect(err).NotTo(gomega.HaveOccurred())
					g.Expect(code).To(gomega.Equal(137))
				}
				code, err := sandboxB.Poll(ctx, nil)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(code).To(gomega.HaveValue(gomega.Equal(137)))
				code, err = sandboxB.Poll(ctx, nil)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(code).To(gomega.HaveValue(gomega.Equal(137)))
				g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
			})
		}
	}
}

func TestPersistentSandboxRefreshErrorMock(t *testing.T) {
	for _, operation := range []string{"poll", "wait", "cancel"} {
		t.Run(operation, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			mock := newGRPCMockClient(t)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			sandbox := mockPersistentSandboxCreate(t, g, mock, nil, nil)
			wantErr := status.Error(codes.NotFound, "unknown")
			grpcmock.HandleUnary(mock, "PersistentSandboxGet", func(*pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
				if operation == "cancel" {
					cancel()
					return pb.PersistentSandboxGetResponse_builder{}.Build(), nil
				}
				return nil, wantErr
			})
			var err error
			if operation == "poll" {
				_, err = sandbox.Poll(ctx, nil)
			} else {
				_, err = sandbox.Wait(ctx, nil)
			}
			if operation == "cancel" {
				g.Expect(err).To(gomega.MatchError(context.Canceled))
			} else {
				g.Expect(err).To(gomega.MatchError(wantErr))
			}
			g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
		})
	}
}

func TestPersistentSandboxDetachedStreamsMock(t *testing.T) {
	for _, streamName := range []string{"stdin", "stdinClose", "stdout", "stderr"} {
		t.Run(streamName, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			mock := newGRPCMockClient(t)
			sandbox, err := mock.Sandboxes.FromID(t.Context(), persistentSandboxID, nil)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(sandbox.Detach()).To(gomega.Succeed())
			switch streamName {
			case "stdin":
				_, err = sandbox.Stdin.Write([]byte("hello"))
			case "stdinClose":
				err = sandbox.Stdin.Close()
			case "stdout":
				_, err = io.ReadAll(sandbox.Stdout)
			case "stderr":
				_, err = io.ReadAll(sandbox.Stderr)
			}
			g.Expect(err).To(gomega.BeAssignableToTypeOf(modal.ClientClosedError{}))
			g.Expect(mock.AssertExhausted()).To(gomega.Succeed())
		})
	}
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
	g.Expect(errors.As(sb.UpdateNetworkPolicy(ctx, &modal.SandboxUpdateNetworkPolicyParams{ //nolint:staticcheck // tests the deprecated method
		OutboundCIDRAllowlist:   &modal.Allowlist{},
		OutboundDomainAllowlist: &modal.Allowlist{},
	}), &invalid)).To(gomega.BeTrue())
	g.Expect(errors.As(sb.SetOutboundPolicy(ctx, &modal.OutboundPolicy{
		Routes: []modal.OutboundRoute{modal.OutboundRouteAny{}},
	}), &invalid)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPauseResumeRequirePersistentSandboxMock(t *testing.T) {
	const v1SandboxID = "sb-nGEijt9WbBMlGrsPH9FOaC"
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()

	for _, id := range []string{v1SandboxID, "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"} {
		sb, err := mock.Sandboxes.FromID(ctx, id, nil)
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		var invalid modal.InvalidError
		g.Expect(errors.As(sb.ExperimentalPause(ctx, nil), &invalid)).To(gomega.BeTrue())
		g.Expect(errors.As(sb.ExperimentalResume(ctx, nil), &invalid)).To(gomega.BeTrue())
	}
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestPersistentSandboxExecReportsStoppedSessionItCannotReachMock(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	mock := newGRPCMockClient(t)
	ctx := t.Context()
	sb := mockPersistentSandboxCreate(t, g, mock, nil, nil)

	handleRouterAccess := func(wantSandboxID string, err error) {
		grpcmock.HandleUnary(mock, "SandboxGetTaskIdV2",
			func(req *pb.SandboxGetTaskIdRequest) (*pb.SandboxGetTaskIdResponse, error) {
				g.Expect(req.GetSandboxId()).To(gomega.Equal(wantSandboxID))
				return pb.SandboxGetTaskIdResponse_builder{TaskId: proto.String("ta-" + wantSandboxID)}.Build(), nil
			})
		grpcmock.HandleUnary(mock, "SandboxGetCommandRouterAccess",
			func(req *pb.SandboxGetCommandRouterAccessRequest) (*pb.SandboxGetCommandRouterAccessResponse, error) {
				g.Expect(req.GetSandboxId()).To(gomega.Equal(wantSandboxID))
				return nil, err
			})
	}
	handleRouterAccess(persistentSessionID, status.Error(codes.FailedPrecondition, "Sandbox has finished"))
	grpcmock.HandleUnary(mock, "PersistentSandboxGet",
		func(req *pb.PersistentSandboxGetRequest) (*pb.PersistentSandboxGetResponse, error) {
			return pb.PersistentSandboxGetResponse_builder{PersistentSandboxId: persistentSandboxID}.Build(), nil
		})
	_, err := sb.Exec(ctx, []string{"echo", "hello"}, nil)
	var paused modal.SandboxPaused
	g.Expect(errors.As(err, &paused)).To(gomega.BeTrue())
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
