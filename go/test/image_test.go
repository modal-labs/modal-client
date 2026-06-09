package test

import (
	"io"
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

func TestImageFromId(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromRegistry("alpine:3.21", nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	imageFromID, err := tc.Images.FromID(ctx, image.ImageID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(imageFromID.ImageID).Should(gomega.Equal(image.ImageID))

	_, err = tc.Images.FromID(ctx, "im-nonexistent", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
}

func TestImageFromName(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock,
		"ImageGetByTag",
		func(req *pb.ImageGetByTagRequest) (*pb.ImageGetByTagResponse, error) {
			g.Expect(req.GetEnvironmentName()).To(gomega.Equal("dev"))
			g.Expect(req.GetTag()).To(gomega.Equal("analytics-runtime:v1"))
			return pb.ImageGetByTagResponse_builder{ImageId: "im-tagged"}.Build(), nil
		},
	)

	image, err := mock.Images.FromName(ctx, "analytics-runtime:v1", &modal.ImageFromNameParams{
		Environment: "dev",
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).To(gomega.Equal("im-tagged"))
	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestImageNamedRefsRejectImageIDNames(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	_, err := mock.Images.FromName(ctx, "im-looks-like-an-id", nil)
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("cannot start with 'im-'")))

	image := mock.Images.FromRegistry("alpine:3.21", nil)
	image.ImageID = "im-built"
	err = image.Publish(ctx, "im-looks-like-an-id", nil)
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("cannot start with 'im-'")))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestImageFromRegistry(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromRegistry("alpine:3.21", nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).Should(gomega.HavePrefix("im-"))
}

func TestImagePublish(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock,
		"ImagePublish",
		func(req *pb.ImagePublishRequest) (*pb.ImagePublishResponse, error) {
			g.Expect(req.GetImageId()).To(gomega.Equal("im-built"))
			g.Expect(req.GetEnvironmentName()).To(gomega.Equal("dev"))
			g.Expect(req.GetIsPublic()).To(gomega.BeFalse())
			g.Expect(req.GetTag()).To(gomega.Equal("analytics-runtime:v1"))
			return pb.ImagePublishResponse_builder{
				ImageId:    req.GetImageId(),
				RevisionId: "ir-01H00000000000000000000000",
			}.Build(), nil
		},
	)

	image := mock.Images.FromRegistry("alpine:3.21", nil)
	image.ImageID = "im-built"
	err := image.Publish(ctx, "analytics-runtime:v1", &modal.ImagePublishParams{
		Environment: "dev",
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestImagePublishRequiresBuiltImage(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	image := mock.Images.FromRegistry("alpine:3.21", nil)
	err := image.Publish(ctx, "analytics-runtime", nil)
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.InvalidError{}))
}

func TestImageFromRegistryWithSecret(t *testing.T) {
	// GCP Artifact Registry also supports auth using username and password, if the username is "_json_key"
	// and the password is the service account JSON blob. See:
	// https://cloud.google.com/artifact-registry/docs/docker/authentication#json-key
	// So we use GCP Artifact Registry to test this too.

	// Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to not trigger
	// the image builder.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")

	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	secret, err := tc.Secrets.FromName(ctx, "libmodal-gcp-artifact-registry-test", &modal.SecretFromNameParams{
		RequiredKeys: []string{"REGISTRY_USERNAME", "REGISTRY_PASSWORD"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromRegistry("us-east1-docker.pkg.dev/modal-prod-367916/private-repo-test/my-image", &modal.ImageFromRegistryParams{Secret: secret}).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).Should(gomega.HavePrefix("im-"))
}

func TestImageFromAwsEcr(t *testing.T) {
	// Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to not trigger
	// the image builder.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")

	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	secret, err := tc.Secrets.FromName(ctx, "libmodal-aws-ecr-test", &modal.SecretFromNameParams{
		RequiredKeys: []string{"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromAwsEcr("459781239556.dkr.ecr.us-east-1.amazonaws.com/ecr-private-registry-test-7522615:python", secret, nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).Should(gomega.HavePrefix("im-"))
}

func TestImageFromGcpArtifactRegistry(t *testing.T) {
	// Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to not trigger
	// the image builder.

	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	secret, err := tc.Secrets.FromName(ctx, "libmodal-gcp-artifact-registry-test", &modal.SecretFromNameParams{
		RequiredKeys: []string{"SERVICE_ACCOUNT_JSON"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromGcpArtifactRegistry("us-east1-docker.pkg.dev/modal-prod-367916/private-repo-test/my-image", secret, nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).Should(gomega.HavePrefix("im-"))
}

func TestCreateSandboxWithImage(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)
	g.Expect(image.ImageID).Should(gomega.BeEmpty())

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	g.Expect(image.ImageID).Should(gomega.HavePrefix("im-"))
}

func TestImageDelete(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromRegistry("alpine:3.13", nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).Should(gomega.HavePrefix("im-"))

	imageFromID, err := tc.Images.FromID(ctx, image.ImageID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(imageFromID.ImageID).Should(gomega.Equal(image.ImageID))

	err = tc.Images.Delete(ctx, image.ImageID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = tc.Images.FromID(ctx, image.ImageID, nil)
	g.Expect(err).Should(gomega.MatchError(gomega.MatchRegexp("Image .+ not found")))

	newImage, err := tc.Images.FromRegistry("alpine:3.13", nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(newImage.ImageID).ShouldNot(gomega.Equal(image.ImageID))

	_, err = tc.Images.FromID(ctx, "im-nonexistent", nil)
	g.Expect(err).Should(gomega.MatchError(gomega.MatchRegexp("Image .+ not found")))
}

func TestDockerfileCommands(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil).DockerfileCommands(
		[]string{"RUN echo hey > /root/hello.txt"},
		nil,
	)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"cat", "/root/hello.txt"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	stdout, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(stdout)).Should(gomega.Equal("hey\n"))
}

func TestDockerfileCommandsEmptyArrayNoOp(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	tc := newTestClient(t)

	image1 := tc.Images.FromRegistry("alpine:3.21", nil)
	image2 := image1.DockerfileCommands([]string{}, nil)
	g.Expect(image2).Should(gomega.BeIdenticalTo(image1))
}

func TestDockerfileCommandsFromNameUsesResolvedImageAsBase(t *testing.T) {
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")

	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock,
		"ImageGetByTag",
		func(req *pb.ImageGetByTagRequest) (*pb.ImageGetByTagResponse, error) {
			g.Expect(req.GetTag()).To(gomega.Equal("analytics-runtime:latest"))
			return pb.ImageGetByTagResponse_builder{ImageId: "im-tagged"}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock,
		"ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM base", "RUN echo layer"}))
			g.Expect(image.GetBaseImages()).To(gomega.HaveLen(1))
			g.Expect(image.GetBaseImages()[0].GetDockerTag()).To(gomega.Equal("base"))
			g.Expect(image.GetBaseImages()[0].GetImageId()).To(gomega.Equal("im-tagged"))

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-layer",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	image, err := mock.Images.FromName(ctx, "analytics-runtime", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	builtImage, err := image.DockerfileCommands([]string{"RUN echo layer"}, nil).
		Build(ctx, &modal.App{AppID: "ap-test"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(builtImage.ImageID).To(gomega.Equal("im-layer"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestDockerfileCommandsFromBuiltRegistryImageDropsRegistryConfig(t *testing.T) {
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")

	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock,
		"ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM private.example.com/app:latest"}))
			g.Expect(image.GetBaseImages()).To(gomega.BeEmpty())
			g.Expect(image.HasImageRegistryConfig()).To(gomega.BeTrue())
			g.Expect(image.GetImageRegistryConfig().GetRegistryAuthType()).To(gomega.Equal(pb.RegistryAuthType_REGISTRY_AUTH_TYPE_STATIC_CREDS))
			g.Expect(image.GetImageRegistryConfig().GetSecretId()).To(gomega.Equal("sc-registry"))

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-private-base",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock,
		"ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM base", "RUN echo layer"}))
			g.Expect(image.GetBaseImages()).To(gomega.HaveLen(1))
			g.Expect(image.GetBaseImages()[0].GetDockerTag()).To(gomega.Equal("base"))
			g.Expect(image.GetBaseImages()[0].GetImageId()).To(gomega.Equal("im-private-base"))
			g.Expect(image.HasImageRegistryConfig()).To(gomega.BeFalse())

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-layer",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	app := &modal.App{AppID: "ap-test"}
	registrySecret := &modal.Secret{SecretID: "sc-registry"}
	baseImage, err := mock.Images.FromRegistry(
		"private.example.com/app:latest",
		&modal.ImageFromRegistryParams{Secret: registrySecret},
	).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	builtImage, err := baseImage.DockerfileCommands([]string{"RUN echo layer"}, nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(builtImage.ImageID).To(gomega.Equal("im-layer"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestDockerfileCommandsChaining(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	secret, err := tc.Secrets.FromMap(ctx, map[string]string{"SECRET": "hello"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil).
		DockerfileCommands([]string{"RUN echo ${SECRET:-unset} > /root/layer1.txt"}, nil).
		DockerfileCommands([]string{"RUN echo ${SECRET:-unset} > /root/layer2.txt"}, &modal.ImageDockerfileCommandsParams{
			Secrets: []*modal.Secret{secret},
		}).
		DockerfileCommands([]string{"RUN echo ${SECRET:-unset} > /root/layer3.txt"}, nil).
		DockerfileCommands([]string{"RUN echo ${SECRET:-unset} > /root/layer4.txt"}, &modal.ImageDockerfileCommandsParams{
			Env: map[string]string{"SECRET": "hello again"},
		})

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{
			"cat",
			"/root/layer1.txt",
			"/root/layer2.txt",
			"/root/layer3.txt",
			"/root/layer4.txt",
		},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	stdout, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(stdout)).Should(gomega.Equal("unset\nhello\nunset\nhello again\n"))
}

func TestDockerfileCommandsCopyCommandValidation(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	validImage := image.DockerfileCommands(
		[]string{"COPY --from=alpine:latest /etc/os-release /tmp/os-release"},
		nil,
	)
	_, err = validImage.Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	invalidImage := image.DockerfileCommands(
		[]string{"COPY ./file.txt /root/"},
		nil,
	)
	_, err = invalidImage.Build(ctx, app, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("COPY commands that copy from local context are not yet supported"))

	runImage := image.DockerfileCommands(
		[]string{"RUN echo 'COPY ./file.txt /root/'"},
		nil,
	)
	_, err = runImage.Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	multiInvalidImage := image.DockerfileCommands(
		[]string{
			"RUN echo hey",
			"copy ./file.txt /root/",
			"RUN echo hey",
		},
		nil,
	)
	_, err = multiInvalidImage.Build(ctx, app, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("COPY commands that copy from local context are not yet supported"))
}

func TestDockerfileCommandsWithOptions(t *testing.T) {

	// Unset `MODAL_IMAGE_BUILDER_VERSION` so that this test uses the image builder version from `EnvironmentGetOrCreate`.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")
	g := gomega.NewWithT(t)
	ctx := t.Context()

	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM alpine:3.21"}))
			g.Expect(image.GetSecretIds()).To(gomega.BeEmpty())
			g.Expect(image.GetBaseImages()).To(gomega.BeEmpty())
			g.Expect(image.GetGpuConfig()).To(gomega.BeNil())
			g.Expect(req.GetForceBuild()).To(gomega.BeFalse())

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-base",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM base", "RUN echo layer1"}))
			g.Expect(image.GetSecretIds()).To(gomega.BeEmpty())
			g.Expect(image.GetBaseImages()).To(gomega.HaveLen(1))
			g.Expect(image.GetBaseImages()[0].GetDockerTag()).To(gomega.Equal("base"))
			g.Expect(image.GetBaseImages()[0].GetImageId()).To(gomega.Equal("im-base"))
			g.Expect(image.GetGpuConfig()).To(gomega.BeNil())
			g.Expect(req.GetForceBuild()).To(gomega.BeFalse())

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-layer1",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM base", "RUN echo layer2"}))
			g.Expect(image.GetSecretIds()).To(gomega.Equal([]string{"sc-test"}))
			g.Expect(image.GetBaseImages()).To(gomega.HaveLen(1))
			g.Expect(image.GetBaseImages()[0].GetDockerTag()).To(gomega.Equal("base"))
			g.Expect(image.GetBaseImages()[0].GetImageId()).To(gomega.Equal("im-layer1"))
			g.Expect(image.GetGpuConfig()).ToNot(gomega.BeNil())
			g.Expect(image.GetGpuConfig().GetType()).To(gomega.Equal(pb.GPUType(0)))
			g.Expect(image.GetGpuConfig().GetCount()).To(gomega.Equal(uint32(1)))
			g.Expect(image.GetGpuConfig().GetGpuType()).To(gomega.Equal("A100"))
			g.Expect(req.GetForceBuild()).To(gomega.BeTrue())

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-layer2",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			image := req.GetImage()
			g.Expect(image.GetDockerfileCommands()).To(gomega.Equal([]string{"FROM base", "RUN echo layer3"}))
			g.Expect(image.GetSecretIds()).To(gomega.BeEmpty())
			g.Expect(image.GetBaseImages()).To(gomega.HaveLen(1))
			g.Expect(image.GetBaseImages()[0].GetDockerTag()).To(gomega.Equal("base"))
			g.Expect(image.GetBaseImages()[0].GetImageId()).To(gomega.Equal("im-layer2"))
			g.Expect(image.GetGpuConfig()).To(gomega.BeNil())
			g.Expect(req.GetForceBuild()).To(gomega.BeTrue())

			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-layer3",
				Result:  pb.GenericResult_builder{Status: pb.GenericResult_GENERIC_STATUS_SUCCESS}.Build(),
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "EnvironmentGetOrCreate",
		func(req *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
			return pb.EnvironmentGetOrCreateResponse_builder{
				EnvironmentId: "en-dev",
				Metadata: pb.EnvironmentMetadata_builder{
					Name: "dev",
					Settings: pb.EnvironmentSettings_builder{
						ImageBuilderVersion: "2025.06",
					}.Build(),
				}.Build(),
			}.Build(), nil
		},
	)

	app := &modal.App{AppID: "ap-test"}
	secret := &modal.Secret{SecretID: "sc-test"}

	builtImage, err := mock.Images.FromRegistry("alpine:3.21", nil).
		DockerfileCommands([]string{"RUN echo layer1"}, nil).
		DockerfileCommands([]string{"RUN echo layer2"}, &modal.ImageDockerfileCommandsParams{
			Secrets:    []*modal.Secret{secret},
			GPU:        "A100",
			ForceBuild: true,
		}).
		DockerfileCommands([]string{"RUN echo layer3"}, &modal.ImageDockerfileCommandsParams{
			ForceBuild: true,
		}).
		Build(ctx, app, nil)

	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(builtImage.ImageID).To(gomega.Equal("im-layer3"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}
