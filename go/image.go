package modal

import (
	"context"
	"fmt"
	"io"
	"strings"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ImageService provides Image related operations.
type ImageService interface {
	FromRegistry(tag string, params *ImageFromRegistryParams) *Image
	FromAwsEcr(tag string, secret *Secret, params *ImageFromAwsEcrParams) *Image
	FromGcpArtifactRegistry(tag string, secret *Secret, params *ImageFromGcpArtifactRegistryParams) *Image
	FromID(ctx context.Context, imageID string, params *ImageFromIDParams) (*Image, error)
	FromName(ctx context.Context, name string, params *ImageFromNameParams) (*Image, error)
	Delete(ctx context.Context, imageID string, params *ImageDeleteParams) error
}

type imageServiceImpl struct{ client *Client }

const defaultImageTag = "latest"

func validateImageName(name string) error {
	if err := checkObjectName(name, "Image"); err != nil {
		return err
	}
	if strings.HasPrefix(name, "im-") {
		return InvalidError{Exception: "Image name cannot start with 'im-' (reserved for image IDs)."}
	}
	return nil
}

func validateImageTag(tag string) error {
	return checkObjectName(tag, "Image tag")
}

// parseNamedImageRef parses an image reference, returning (namespacePrefix, nameTag).
// If the name contains a '/', the part before the last '/' is extracted as a
// namespace prefix (intended for environment/name or workspace/env/name syntax).
// The actual image name (after the last '/') is validated as a standard image name.
func parseNamedImageRef(value string) (string, string, error) {
	imageName, tag, found := strings.Cut(value, ":")
	if !found {
		tag = defaultImageTag
	}

	prefix := ""
	if lastSlash := strings.LastIndex(imageName, "/"); lastSlash != -1 {
		prefix = imageName[:lastSlash]
		after := imageName[lastSlash+1:]
		if prefix == "" {
			return "", "", InvalidError{Exception: "Invalid Image name: '/' prefix must be non-empty."}
		}
		if after == "" {
			return "", "", InvalidError{Exception: "Invalid Image name: name after '/' must be non-empty."}
		}
		imageName = after
	}

	if err := validateImageName(imageName); err != nil {
		return "", "", err
	}
	if err := validateImageTag(tag); err != nil {
		return "", "", err
	}

	fullName := imageName
	if prefix != "" {
		fullName = prefix + "/" + imageName
	}
	return prefix, fullName + ":" + tag, nil
}

// ImageDockerfileCommandsParams are options for Image.DockerfileCommands().
type ImageDockerfileCommandsParams struct {
	// Environment variables to set in the build environment.
	Env map[string]string

	// Secrets that will be made available as environment variables to this layer's build environment.
	Secrets []*Secret

	// GPU reservation for this layer's build environment (e.g. "A100", "T4:2", "A100-80GB:4").
	GPU string

	// Ignore cached builds for this layer, similar to 'docker build --no-cache'.
	ForceBuild bool
}

// layer represents a single image layer with its build configuration.
type layer struct {
	commands   []string
	env        map[string]string
	secrets    []*Secret
	gpu        string
	forceBuild bool
}

// Image represents a Modal Image, which can be used to create Sandboxes.
type Image struct {
	ImageID string

	// buildRegistryConfig, when non-nil, hydrates the registry-auth Secret (if
	// any) and returns the ImageRegistryConfig for the base build layer. It runs
	// at Build time rather than when the Image is constructed, since the Secret
	// may be lazy (created via FromMap). nil when not pulling from a private registry.
	buildRegistryConfig func(ctx context.Context, client *Client) (*pb.ImageRegistryConfig, error)
	tag                 string
	// baseImageID is the image ID of the parent image layer to use as FROM base.
	baseImageID string
	layers      []layer

	client *Client
}

// ImageFromRegistryParams are options for creating an Image from a registry.
type ImageFromRegistryParams struct {
	Secret *Secret
}

// ImageFromAwsEcrParams are options for ImageService.FromAwsEcr.
type ImageFromAwsEcrParams struct{}

// ImageFromGcpArtifactRegistryParams are options for ImageService.FromGcpArtifactRegistry.
type ImageFromGcpArtifactRegistryParams struct{}

// ImageFromIDParams are options for ImageService.FromID.
type ImageFromIDParams struct{}

// ImageFromNameParams are options for ImageService.FromName.
type ImageFromNameParams struct {
	Environment string
}

// ImageBuildParams are options for Image.Build.
type ImageBuildParams struct{}

// ImagePublishParams are options for Image.Publish.
type ImagePublishParams struct {
	Environment string
}

// registryConfigBuilder returns a closure that hydrates the registry-auth Secret
// (if any) and builds the ImageRegistryConfig. It runs at Build time so that lazy
// Secrets (created via FromMap) are resolved against the build's client.
func registryConfigBuilder(authType pb.RegistryAuthType, secret *Secret) func(ctx context.Context, client *Client) (*pb.ImageRegistryConfig, error) {
	return func(ctx context.Context, client *Client) (*pb.ImageRegistryConfig, error) {
		secretID := ""
		if secret != nil {
			if err := hydrateSecrets(ctx, client, []*Secret{secret}); err != nil {
				return nil, err
			}
			secretID = secret.SecretID
		}
		return pb.ImageRegistryConfig_builder{
			RegistryAuthType: authType,
			SecretId:         secretID,
		}.Build(), nil
	}
}

// FromRegistry builds a Modal Image from a public or private image registry without any changes.
func (s *imageServiceImpl) FromRegistry(tag string, params *ImageFromRegistryParams) *Image {
	image := &Image{
		ImageID: "",
		tag:     tag,
		layers:  []layer{{}},
		client:  s.client,
	}
	if params != nil && params.Secret != nil {
		image.buildRegistryConfig = registryConfigBuilder(pb.RegistryAuthType_REGISTRY_AUTH_TYPE_STATIC_CREDS, params.Secret)
	}
	return image
}

// FromAwsEcr creates an Image from an AWS ECR tag
func (s *imageServiceImpl) FromAwsEcr(tag string, secret *Secret, params *ImageFromAwsEcrParams) *Image {
	return &Image{
		ImageID:             "",
		buildRegistryConfig: registryConfigBuilder(pb.RegistryAuthType_REGISTRY_AUTH_TYPE_AWS, secret),
		tag:                 tag,
		layers:              []layer{{}},
		client:              s.client,
	}
}

// FromGcpArtifactRegistry creates an Image from a GCP Artifact Registry tag.
func (s *imageServiceImpl) FromGcpArtifactRegistry(tag string, secret *Secret, params *ImageFromGcpArtifactRegistryParams) *Image {
	return &Image{
		ImageID:             "",
		buildRegistryConfig: registryConfigBuilder(pb.RegistryAuthType_REGISTRY_AUTH_TYPE_GCP, secret),
		tag:                 tag,
		layers:              []layer{{}},
		client:              s.client,
	}
}

// FromID looks up an Image from an ID
func (s *imageServiceImpl) FromID(ctx context.Context, imageID string, params *ImageFromIDParams) (*Image, error) {
	resp, err := s.client.cpClient.ImageFromId(
		ctx,
		pb.ImageFromIdRequest_builder{
			ImageId: imageID,
		}.Build(),
	)
	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Image '%s' not found", imageID)}
	}
	if err != nil {
		return nil, err
	}

	return &Image{
		ImageID: resp.GetImageId(),
		layers:  []layer{{}},
		client:  s.client,
	}, nil
}

// FromName references a named Image that was previously published.
// The name may include a tag as name:tag; if no tag is included, :latest is used.
func (s *imageServiceImpl) FromName(ctx context.Context, name string, params *ImageFromNameParams) (*Image, error) {
	if params == nil {
		params = &ImageFromNameParams{}
	}
	namespacePrefix, tag, err := parseNamedImageRef(name)
	if err != nil {
		return nil, err
	}

	var environmentName string
	if namespacePrefix != "" {
		if params.Environment != "" {
			return nil, InvalidError{Exception: "Cannot specify 'Environment' when the image name contains a '/'."}
		}
		environmentName = ""
	} else {
		environmentName = firstNonEmpty(params.Environment, s.client.profile.Environment)
	}

	resp, err := s.client.cpClient.ImageGetByTag(ctx, pb.ImageGetByTagRequest_builder{
		EnvironmentName: environmentName,
		Tag:             tag,
	}.Build())
	if st, ok := status.FromError(err); ok && st.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Image '%s' not found", tag)}
	}
	if err != nil {
		return nil, err
	}

	return &Image{
		ImageID: resp.GetImageId(),
		layers:  []layer{{}},
		client:  s.client,
	}, nil
}

// DockerfileCommands extends an image with arbitrary Dockerfile-like commands.
//
// Each call creates a new Image layer that will be built sequentially.
// The provided options apply only to this layer.
func (image *Image) DockerfileCommands(commands []string, params *ImageDockerfileCommandsParams) *Image {
	if len(commands) == 0 {
		return image
	}

	if params == nil {
		params = &ImageDockerfileCommandsParams{}
	}

	newLayer := layer{
		commands:   append([]string{}, commands...),
		env:        params.Env,
		secrets:    params.Secrets,
		gpu:        params.GPU,
		forceBuild: params.ForceBuild,
	}

	baseImageID := image.baseImageID
	newLayers := append([]layer{}, image.layers...)
	if image.ImageID != "" {
		baseImageID = image.ImageID
		newLayers = []layer{}
	}
	newLayers = append(newLayers, newLayer)

	return &Image{
		ImageID:             "",
		tag:                 image.tag,
		baseImageID:         baseImageID,
		buildRegistryConfig: image.buildRegistryConfig,
		layers:              newLayers,
		client:              image.client,
	}
}

func validateDockerfileCommands(commands []string) error {
	for _, command := range commands {
		trimmed := strings.ToUpper(strings.TrimSpace(command))
		if strings.HasPrefix(trimmed, "COPY ") && !strings.HasPrefix(trimmed, "COPY --FROM=") {
			return InvalidError{"COPY commands that copy from local context are not yet supported."}
		}
	}
	return nil
}

// waitForBuildIteration performs a single iteration of waiting for an image build to complete.
// It streams build updates and returns either a result (if build completes) or nil (if the stream
// ends without a result, requiring another iteration).
func (image *Image) waitForBuildIteration(ctx context.Context, imageID string, lastEntryID string) (*pb.GenericResult, string, error) {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := image.client.cpClient.ImageJoinStreaming(streamCtx, pb.ImageJoinStreamingRequest_builder{
		ImageId:     imageID,
		Timeout:     55,
		LastEntryId: lastEntryID,
	}.Build())
	if err != nil {
		return nil, lastEntryID, err
	}

	for {
		item, err := stream.Recv()
		if err == io.EOF {
			return nil, lastEntryID, nil
		} else if err != nil {
			return nil, lastEntryID, err
		}

		if item.GetEntryId() != "" {
			lastEntryID = item.GetEntryId()
		}

		// Ignore all log lines and progress updates.
		res := item.GetResult()
		if res == nil || res.GetStatus() == pb.GenericResult_GENERIC_STATUS_UNSPECIFIED {
			continue
		}

		return res, lastEntryID, nil
	}
}

// Build eagerly builds an Image on Modal.
func (image *Image) Build(ctx context.Context, app *App, params *ImageBuildParams) (*Image, error) {
	// Image is already hyrdated
	if image.ImageID != "" {
		return image, nil
	}

	image.client.logger.DebugContext(ctx, "Building image", "app_id", app.AppID)

	for _, currentLayer := range image.layers {
		if err := validateDockerfileCommands(currentLayer.commands); err != nil {
			return nil, err
		}
	}

	currentImageID := image.baseImageID
	imageBuilderVersion, err := image.client.getImageBuilderVersion(ctx, app.Environment)
	if err != nil {
		return nil, err
	}

	image.client.logger.DebugContext(ctx, "Image build", "app_id", app.AppID, "image_builder_version", imageBuilderVersion, "environment_name", app.Environment)

	for i, currentLayer := range image.layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		mergedSecrets, err := mergeEnvIntoSecrets(ctx, image.client, &currentLayer.env, &currentLayer.secrets)
		if err != nil {
			return nil, err
		}

		if err := hydrateSecrets(ctx, image.client, mergedSecrets); err != nil {
			return nil, err
		}

		var secretIds []string
		for _, secret := range mergedSecrets {
			secretIds = append(secretIds, secret.SecretID)
		}

		var gpuConfig *pb.GPUConfig
		if currentLayer.gpu != "" {
			var err error
			gpuConfig, err = parseGPUConfig(currentLayer.gpu)
			if err != nil {
				return nil, err
			}
		}

		var dockerfileCommands []string
		var baseImages []*pb.BaseImage

		if i == 0 && currentImageID != "" {
			dockerfileCommands = append([]string{"FROM base"}, currentLayer.commands...)
			baseImages = []*pb.BaseImage{pb.BaseImage_builder{
				DockerTag: "base",
				ImageId:   currentImageID,
			}.Build()}
		} else if i == 0 {
			dockerfileCommands = append([]string{fmt.Sprintf("FROM %s", image.tag)}, currentLayer.commands...)
			baseImages = []*pb.BaseImage{}
		} else {
			dockerfileCommands = append([]string{"FROM base"}, currentLayer.commands...)
			baseImages = []*pb.BaseImage{pb.BaseImage_builder{
				DockerTag: "base",
				ImageId:   currentImageID,
			}.Build()}
		}

		var imageRegistryConfig *pb.ImageRegistryConfig
		if i == 0 && currentImageID == "" && image.buildRegistryConfig != nil {
			imageRegistryConfig, err = image.buildRegistryConfig(ctx, image.client)
			if err != nil {
				return nil, err
			}
		}

		resp, err := image.client.cpClient.ImageGetOrCreate(
			ctx,
			pb.ImageGetOrCreateRequest_builder{
				AppId: app.AppID,
				Image: pb.Image_builder{
					DockerfileCommands:  dockerfileCommands,
					ImageRegistryConfig: imageRegistryConfig,
					SecretIds:           secretIds,
					GpuConfig:           gpuConfig,
					ContextFiles:        []*pb.ImageContextFile{},
					BaseImages:          baseImages,
				}.Build(),
				BuilderVersion: imageBuilderVersion,
				ForceBuild:     currentLayer.forceBuild,
			}.Build(),
		)
		if err != nil {
			return nil, err
		}

		result := resp.GetResult()

		if result == nil || result.GetStatus() == pb.GenericResult_GENERIC_STATUS_UNSPECIFIED {
			// Not built or in the process of building - wait for build
			lastEntryID := ""
			for result == nil {
				if err := ctx.Err(); err != nil {
					return nil, err
				}

				var err error
				result, lastEntryID, err = image.waitForBuildIteration(ctx, resp.GetImageId(), lastEntryID)
				if err != nil {
					return nil, err
				}
			}
		}

		switch result.GetStatus() {
		case pb.GenericResult_GENERIC_STATUS_FAILURE:
			return nil, RemoteError{fmt.Sprintf("Image build for %s failed with the exception:\n%s", resp.GetImageId(), result.GetException())}
		case pb.GenericResult_GENERIC_STATUS_TERMINATED:
			return nil, RemoteError{fmt.Sprintf("Image build for %s terminated due to external shut-down. Please try again.", resp.GetImageId())}
		case pb.GenericResult_GENERIC_STATUS_TIMEOUT:
			return nil, RemoteError{fmt.Sprintf("Image build for %s timed out. Please try again with a larger timeout parameter.", resp.GetImageId())}
		case pb.GenericResult_GENERIC_STATUS_SUCCESS:
			// Success, do nothing
		default:
			return nil, RemoteError{fmt.Sprintf("Image build for %s failed with unknown status: %s", resp.GetImageId(), result.GetStatus())}
		}

		// The new image becomes the base for the next layer
		currentImageID = resp.GetImageId()
	}

	image.ImageID = currentImageID
	image.client.logger.DebugContext(ctx, "Image build completed", "image_id", currentImageID)
	return image, nil
}

// Publish publishes this built Image under a stable name and tag.
// The name may include a tag as name:tag; if no tag is included, :latest is used.
func (image *Image) Publish(ctx context.Context, name string, params *ImagePublishParams) error {
	if params == nil {
		params = &ImagePublishParams{}
	}
	namespacePrefix, tag, err := parseNamedImageRef(name)
	if err != nil {
		return err
	}
	if image.ImageID == "" {
		return InvalidError{Exception: "Cannot publish an image that has not been built yet. Call Build() first."}
	}

	var environmentName string
	if namespacePrefix != "" {
		if params.Environment != "" {
			return InvalidError{Exception: "Cannot specify 'Environment' when the image name contains a '/'."}
		}
		environmentName = ""
	} else {
		environmentName = firstNonEmpty(params.Environment, image.client.profile.Environment)
	}

	_, err = image.client.cpClient.ImagePublish(ctx, pb.ImagePublishRequest_builder{
		ImageId:         image.ImageID,
		EnvironmentName: environmentName,
		AllowPublic:     false,
		Tag:             tag,
	}.Build())
	return err
}

// ImageDeleteParams are options for deleting an Image.
type ImageDeleteParams struct{}

// Delete deletes an Image by ID.
//
// Deletion is irreversible and will prevent Functions/Sandboxes from using the Image.
//
// Note: When building an Image, each chained method call will create an
// intermediate Image layer, each with its own ID. Deleting an Image will not
// delete any of its intermediate layers, only the image identified by the
// provided ID.
func (s *imageServiceImpl) Delete(ctx context.Context, imageID string, params *ImageDeleteParams) error {
	_, err := s.client.cpClient.ImageDelete(ctx, pb.ImageDeleteRequest_builder{ImageId: imageID}.Build())
	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return NotFoundError{fmt.Sprintf("Image '%s' not found", imageID)}
	}
	return err
}
