package modal

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

type environmentServiceMockClient struct {
	pb.ModalClientClient
	lookup    func(context.Context, *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error)
	list      func(context.Context, *pb.AppListRequest) (*pb.AppListResponse, error)
	create    func(context.Context, *pb.EnvironmentCreateRequest) (*emptypb.Empty, error)
	deleteEnv func(context.Context, *pb.EnvironmentDeleteRequest) (*emptypb.Empty, error)
	listEnvs  func(context.Context, *emptypb.Empty) (*pb.EnvironmentListResponse, error)
}

func (m *environmentServiceMockClient) EnvironmentCreate(ctx context.Context, req *pb.EnvironmentCreateRequest, _ ...grpc.CallOption) (*emptypb.Empty, error) {
	return m.create(ctx, req)
}

func (m *environmentServiceMockClient) EnvironmentDelete(ctx context.Context, req *pb.EnvironmentDeleteRequest, _ ...grpc.CallOption) (*emptypb.Empty, error) {
	return m.deleteEnv(ctx, req)
}

func (m *environmentServiceMockClient) EnvironmentList(ctx context.Context, req *emptypb.Empty, _ ...grpc.CallOption) (*pb.EnvironmentListResponse, error) {
	return m.listEnvs(ctx, req)
}

func (m *environmentServiceMockClient) EnvironmentGetOrCreate(ctx context.Context, req *pb.EnvironmentGetOrCreateRequest, _ ...grpc.CallOption) (*pb.EnvironmentGetOrCreateResponse, error) {
	return m.lookup(ctx, req)
}

func (m *environmentServiceMockClient) AppList(ctx context.Context, req *pb.AppListRequest, _ ...grpc.CallOption) (*pb.AppListResponse, error) {
	return m.list(ctx, req)
}

func newEnvironmentServiceTestClient(t *testing.T, mock pb.ModalClientClient) *Client {
	t.Helper()
	client, err := NewClientWithOptions(&ClientParams{
		ControlPlaneClient: mock,
		TokenID:            "test-token-id", TokenSecret: "test-token-secret",
		Environment: "dev", Config: &config{},
	})
	NewWithT(t).Expect(err).ShouldNot(HaveOccurred())
	t.Cleanup(client.Close)
	return client
}

func TestEnvironmentFromName(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name         string
		params       *EnvironmentFromNameParams
		creationType pb.ObjectCreationType
	}{
		{"prod", nil, pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED},
		{"prod.v2", &EnvironmentFromNameParams{}, pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED},
		{"prod", &EnvironmentFromNameParams{CreateIfMissing: true}, pb.ObjectCreationType_OBJECT_CREATION_TYPE_CREATE_IF_MISSING},
	} {
		t.Run(fmt.Sprintf("%s/%v", tc.name, tc.params), func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			mock := &environmentServiceMockClient{lookup: func(_ context.Context, req *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
				g.Expect(req.GetDeploymentName()).To(Equal(tc.name))
				g.Expect(req.GetObjectCreationType()).To(Equal(tc.creationType))
				return pb.EnvironmentGetOrCreateResponse_builder{
					EnvironmentId: "en-resolved",
					Metadata:      pb.EnvironmentMetadata_builder{Name: "prod"}.Build(),
				}.Build(), nil
			}}
			client := newEnvironmentServiceTestClient(t, mock)
			env, err := client.Environments.FromName(t.Context(), tc.name, tc.params)
			g.Expect(err).ShouldNot(HaveOccurred())
			g.Expect(env.EnvironmentID).To(Equal("en-resolved"))
			g.Expect(env.Name).To(Equal("prod"))
			g.Expect(env.Apps).NotTo(BeNil())
		})
	}
}

func TestEnvironmentFromNameErrors(t *testing.T) {
	t.Parallel()
	for _, code := range []codes.Code{codes.NotFound, codes.PermissionDenied} {
		t.Run(code.String(), func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			rpcErr := status.Error(code, "lookup failed")
			mock := &environmentServiceMockClient{lookup: func(context.Context, *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
				return nil, rpcErr
			}}
			client := newEnvironmentServiceTestClient(t, mock)
			env, err := client.Environments.FromName(t.Context(), "prod", nil)
			g.Expect(env).To(BeNil())
			if code == codes.NotFound {
				g.Expect(err).To(BeAssignableToTypeOf(NotFoundError{}))
			} else {
				g.Expect(err).To(Equal(rpcErr))
			}
		})
	}
}

func TestEnvironmentAppsList(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	listCalls := 0
	mock := &environmentServiceMockClient{
		lookup: func(context.Context, *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
			return pb.EnvironmentGetOrCreateResponse_builder{
				EnvironmentId: "en-prod", Metadata: pb.EnvironmentMetadata_builder{Name: "prod"}.Build(),
			}.Build(), nil
		},
		list: func(_ context.Context, req *pb.AppListRequest) (*pb.AppListResponse, error) {
			listCalls++
			g.Expect(req.GetEnvironmentName()).To(Equal("prod"))
			return pb.AppListResponse_builder{Apps: []*pb.AppListResponse_AppListItem{
				pb.AppListResponse_AppListItem_builder{AppId: "ap-deployed", Name: "libmodal-test", State: pb.AppState_APP_STATE_DEPLOYED}.Build(),
				pb.AppListResponse_AppListItem_builder{AppId: "ap-stopped", State: pb.AppState_APP_STATE_STOPPED}.Build(),
				pb.AppListResponse_AppListItem_builder{AppId: "ap-ephemeral", Description: "libmodal-test", State: pb.AppState_APP_STATE_EPHEMERAL}.Build(),
				pb.AppListResponse_AppListItem_builder{AppId: "ap-disabled", State: pb.AppState_APP_STATE_DISABLED}.Build(),
				pb.AppListResponse_AppListItem_builder{AppId: "ap-stopping", Name: "libmodal-test-support", State: pb.AppState_APP_STATE_STOPPING}.Build(),
			}}.Build(), nil
		},
	}
	client := newEnvironmentServiceTestClient(t, mock)
	env, err := client.Environments.FromName(t.Context(), "alias", nil)
	g.Expect(err).ShouldNot(HaveOccurred())
	apps, err := env.Apps.List(t.Context())
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(listCalls).To(Equal(0))
	var got []*App
	for app, err := range apps {
		g.Expect(err).ShouldNot(HaveOccurred())
		got = append(got, app)
	}
	var identities [][3]string
	for _, app := range got {
		identities = append(identities, [3]string{app.AppID, app.Name, app.Environment})
	}
	g.Expect(identities).To(Equal([][3]string{
		{"ap-deployed", "libmodal-test", "prod"},
		{"ap-ephemeral", "", "prod"},
		{"ap-stopping", "libmodal-test-support", "prod"},
	}))
	g.Expect(listCalls).To(Equal(1))
	apps, err = env.Apps.List(t.Context())
	g.Expect(err).ShouldNot(HaveOccurred())
	count := 0
	for app, err := range apps {
		g.Expect(err).ShouldNot(HaveOccurred())
		g.Expect(app.AppID).To(Equal("ap-deployed"))
		count++
		break
	}
	g.Expect(count).To(Equal(1))
	g.Expect(listCalls).To(Equal(2))
}

func TestEnvironmentAppsListEmptyAndErrors(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		err    error
		cancel bool
	}{
		{name: "empty"},
		{name: "rpc-error", err: status.Error(codes.PermissionDenied, "access denied")},
		{name: "cancelled", err: context.Canceled, cancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			calls := 0
			mock := &environmentServiceMockClient{list: func(context.Context, *pb.AppListRequest) (*pb.AppListResponse, error) {
				calls++
				return pb.AppListResponse_builder{}.Build(), tc.err
			}}
			client := newEnvironmentServiceTestClient(t, mock)
			service := &environmentAppsServiceImpl{client: client, environmentName: "prod"}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc.cancel {
				cancel()
			}
			apps, err := service.List(ctx)
			g.Expect(err).ShouldNot(HaveOccurred())
			count := 0
			for app, err := range apps {
				g.Expect(app).To(BeNil())
				g.Expect(err).To(Equal(tc.err))
				count++
			}
			if tc.err == nil {
				g.Expect(count).To(Equal(0))
			} else {
				g.Expect(count).To(Equal(1))
			}
			if tc.cancel {
				g.Expect(calls).To(Equal(0))
			} else {
				g.Expect(calls).To(Equal(1))
			}
		})
	}
}

func TestEnvironmentNameValidation(t *testing.T) {
	t.Parallel()
	invalid := []string{"", "en-prod", "EN-prod", "En-prod", "_prod", "-prod", ".prod", "prod/v2", "prod:v2", "prod dev", "prod\n", "pröd", strings.Repeat("a", 65)}
	for _, name := range invalid {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			client := newEnvironmentServiceTestClient(t, &environmentServiceMockClient{})
			env, err := client.Environments.FromName(t.Context(), name, nil)
			g.Expect(env).To(BeNil())
			g.Expect(err).To(BeAssignableToTypeOf(InvalidError{}))
			g.Expect(client.Environments.Objects.Create(t.Context(), name, nil)).To(BeAssignableToTypeOf(InvalidError{}))
		})
	}
	for _, name := range []string{"a", "9prod", "Prod_v2-dev", strings.Repeat("a", 64)} {
		g := NewWithT(t)
		g.Expect(checkEnvironmentName(name, true)).To(Succeed())
		g.Expect(checkEnvironmentName(name, false)).To(Succeed())
	}
	g := NewWithT(t)
	g.Expect(checkEnvironmentName("prod.v2", true)).To(Succeed())
	g.Expect(checkEnvironmentName("prod.v2", false)).To(BeAssignableToTypeOf(InvalidError{}))
}

func TestEnvironmentObjectsCreate(t *testing.T) {
	t.Parallel()
	for _, params := range []*EnvironmentCreateParams{nil, {Restricted: true, DefaultRole: "viewer"}} {
		t.Run(fmt.Sprint(params), func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			calls := 0
			mock := &environmentServiceMockClient{create: func(_ context.Context, req *pb.EnvironmentCreateRequest) (*emptypb.Empty, error) {
				calls++
				g.Expect(req.GetName()).To(Equal("prod"))
				if params == nil {
					g.Expect(req.GetIsManaged()).To(BeFalse())
					g.Expect(req.GetDefaultMemberRoleStr()).To(BeEmpty())
				} else {
					g.Expect(req.GetIsManaged()).To(BeTrue())
					g.Expect(req.GetDefaultMemberRoleStr()).To(Equal("viewer"))
				}
				return &emptypb.Empty{}, nil
			}}
			client := newEnvironmentServiceTestClient(t, mock)
			g.Expect(client.Environments.Objects.Create(t.Context(), "prod", params)).To(Succeed())
			g.Expect(calls).To(Equal(1))
		})
	}
}

func TestEnvironmentObjectsList(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	calls := 0
	mock := &environmentServiceMockClient{
		listEnvs: func(_ context.Context, _ *emptypb.Empty) (*pb.EnvironmentListResponse, error) {
			calls++
			return pb.EnvironmentListResponse_builder{Items: []*pb.EnvironmentListItem{
				pb.EnvironmentListItem_builder{EnvironmentId: "en-prod", Name: "prod"}.Build(),
				pb.EnvironmentListItem_builder{EnvironmentId: "en-dev", Name: "dev"}.Build(),
			}}.Build(), nil
		},
		list: func(_ context.Context, req *pb.AppListRequest) (*pb.AppListResponse, error) {
			return pb.AppListResponse_builder{Apps: []*pb.AppListResponse_AppListItem{
				pb.AppListResponse_AppListItem_builder{
					AppId: "ap-listed", Name: "libmodal-test", State: pb.AppState_APP_STATE_DEPLOYED,
					Metadata: pb.AppHandleMetadata_builder{Description: "listed app", Functions: map[string]string{"echo": "fu-echo"}}.Build(),
				}.Build(),
			}}.Build(), nil
		},
	}
	client := newEnvironmentServiceTestClient(t, mock)
	environments, err := client.Environments.Objects.List(t.Context())
	g.Expect(err).ShouldNot(HaveOccurred())
	g.Expect(calls).To(Equal(0))
	var names []string
	for env, err := range environments {
		g.Expect(err).ShouldNot(HaveOccurred())
		names = append(names, env.Name)
		g.Expect(env.EnvironmentID).To(Equal("en-" + env.Name))
		apps, err := env.Apps.List(t.Context())
		g.Expect(err).ShouldNot(HaveOccurred())
		for app, err := range apps {
			g.Expect(err).ShouldNot(HaveOccurred())
			g.Expect(app.Environment).To(Equal(env.Name))
			info, err := app.Info(t.Context(), nil)
			g.Expect(err).ShouldNot(HaveOccurred())
			g.Expect(info.Description).To(Equal("listed app"))
			g.Expect(info.Functions).To(Equal(map[string]string{"echo": "fu-echo"}))
		}
	}
	g.Expect(names).To(Equal([]string{"prod", "dev"}))
	g.Expect(calls).To(Equal(1))
	environments, err = client.Environments.Objects.List(t.Context())
	g.Expect(err).ShouldNot(HaveOccurred())
	count := 0
	for _, err := range environments {
		g.Expect(err).ShouldNot(HaveOccurred())
		count++
		break
	}
	g.Expect(count).To(Equal(1))
	g.Expect(calls).To(Equal(2))
}

func TestEnvironmentObjectsListEmptyAndErrors(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name      string
		rpcErr    error
		cancelled bool
	}{
		{name: "empty"},
		{name: "error", rpcErr: status.Error(codes.PermissionDenied, "denied")},
		{name: "cancelled", rpcErr: context.Canceled, cancelled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := NewWithT(t)
			calls := 0
			mock := &environmentServiceMockClient{listEnvs: func(context.Context, *emptypb.Empty) (*pb.EnvironmentListResponse, error) {
				calls++
				return pb.EnvironmentListResponse_builder{}.Build(), tc.rpcErr
			}}
			client := newEnvironmentServiceTestClient(t, mock)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc.cancelled {
				cancel()
			}
			environments, err := client.Environments.Objects.List(ctx)
			g.Expect(err).ShouldNot(HaveOccurred())
			count := 0
			for env, err := range environments {
				count++
				g.Expect(env).To(BeNil())
				g.Expect(err).To(Equal(tc.rpcErr))
			}
			if tc.rpcErr == nil {
				g.Expect(count).To(Equal(0))
			} else {
				g.Expect(count).To(Equal(1))
			}
			if tc.cancelled {
				g.Expect(calls).To(Equal(0))
			} else {
				g.Expect(calls).To(Equal(1))
			}
		})
	}
}

func TestEnvironmentObjectsDeleteAndErrors(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	mock := &environmentServiceMockClient{deleteEnv: func(_ context.Context, req *pb.EnvironmentDeleteRequest) (*emptypb.Empty, error) {
		g.Expect(req.GetName()).To(Equal("prod"))
		return &emptypb.Empty{}, nil
	}}
	client := newEnvironmentServiceTestClient(t, mock)
	g.Expect(client.Environments.Objects.Delete(t.Context(), "prod")).To(Succeed())
	for _, code := range []codes.Code{codes.NotFound, codes.PermissionDenied} {
		rpcErr := status.Error(code, "delete failed")
		mock.deleteEnv = func(context.Context, *pb.EnvironmentDeleteRequest) (*emptypb.Empty, error) { return nil, rpcErr }
		err := client.Environments.Objects.Delete(t.Context(), "prod")
		if code == codes.NotFound {
			g.Expect(err).To(BeAssignableToTypeOf(NotFoundError{}))
		} else {
			g.Expect(err).To(Equal(rpcErr))
		}
	}
	for _, code := range []codes.Code{codes.AlreadyExists, codes.InvalidArgument, codes.PermissionDenied} {
		rpcErr := status.Error(code, "create failed")
		mock.create = func(context.Context, *pb.EnvironmentCreateRequest) (*emptypb.Empty, error) { return nil, rpcErr }
		err := client.Environments.Objects.Create(t.Context(), "prod", nil)
		switch code {
		case codes.AlreadyExists:
			g.Expect(err).To(BeAssignableToTypeOf(AlreadyExistsError{}))
		case codes.InvalidArgument:
			g.Expect(err).To(BeAssignableToTypeOf(InvalidError{}))
		default:
			g.Expect(err).To(Equal(rpcErr))
		}
	}
}

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

func TestEnvironmentFromContext(t *testing.T) {
	for _, tc := range []struct {
		configured   string
		expectedName string
	}{
		{"dev", "dev"},
		{"", "main"},
	} {
		t.Run(fmt.Sprintf("configured=%q", tc.configured), func(t *testing.T) {
			t.Setenv("MODAL_ENVIRONMENT", "")
			g := NewWithT(t)
			mock := &environmentServiceMockClient{lookup: func(_ context.Context, req *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
				g.Expect(req.GetDeploymentName()).To(Equal(tc.configured))
				g.Expect(req.GetObjectCreationType()).To(Equal(pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED))
				return pb.EnvironmentGetOrCreateResponse_builder{
					EnvironmentId: "en-" + tc.expectedName,
					Metadata:      pb.EnvironmentMetadata_builder{Name: tc.expectedName}.Build(),
				}.Build(), nil
			}}
			client, err := NewClientWithOptions(&ClientParams{
				ControlPlaneClient: mock,
				TokenID:            "test-token-id", TokenSecret: "test-token-secret",
				Environment: tc.configured, Config: &config{},
			})
			g.Expect(err).ShouldNot(HaveOccurred())
			t.Cleanup(client.Close)
			env, err := client.Environments.FromContext(t.Context())
			g.Expect(err).ShouldNot(HaveOccurred())
			g.Expect(env.EnvironmentID).To(Equal("en-" + tc.expectedName))
			g.Expect(env.Name).To(Equal(tc.expectedName))
			g.Expect(env.Apps).NotTo(BeNil())
		})
	}
}

func TestEnvironmentFromContextNotFound(t *testing.T) {
	t.Parallel()
	g := NewWithT(t)
	mock := &environmentServiceMockClient{lookup: func(context.Context, *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
		return nil, status.Error(codes.NotFound, "lookup failed")
	}}
	client := newEnvironmentServiceTestClient(t, mock)
	env, err := client.Environments.FromContext(t.Context())
	g.Expect(env).To(BeNil())
	g.Expect(err).To(Equal(NotFoundError{"Environment 'dev' not found"}))
}
