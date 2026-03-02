package modal

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
)

// CloudBucketMountService provides CloudBucketMount related operations.
type CloudBucketMountService interface {
	New(bucketName string, params *CloudBucketMountParams) (*CloudBucketMount, error)
}

type cloudBucketMountServiceImpl struct{ client *Client }

// CloudBucketMount provides access to cloud storage buckets within Modal Functions.
type CloudBucketMount struct {
	BucketName        string
	Secret            *Secret
	ReadOnly          bool
	RequesterPays     bool
	BucketEndpointURL *string
	KeyPrefix         *string
	OidcAuthRoleArn   *string
	bucketType        pb.CloudBucketMount_BucketType
}

// CloudBucketMountParams are options for creating a CloudBucketMount.
type CloudBucketMountParams struct {
	Secret            *Secret
	ReadOnly          bool
	RequesterPays     bool
	BucketEndpointURL *string
	KeyPrefix         *string
	OidcAuthRoleArn   *string
}

// New creates a new CloudBucketMount.
func (s *cloudBucketMountServiceImpl) New(bucketName string, params *CloudBucketMountParams) (*CloudBucketMount, error) {
	ctx := context.Background()
	if params == nil {
		params = &CloudBucketMountParams{}
	}

	mount := &CloudBucketMount{
		BucketName:        bucketName,
		Secret:            params.Secret,
		ReadOnly:          params.ReadOnly,
		RequesterPays:     params.RequesterPays,
		BucketEndpointURL: params.BucketEndpointURL,
		KeyPrefix:         params.KeyPrefix,
		OidcAuthRoleArn:   params.OidcAuthRoleArn,
	}

	if mount.BucketEndpointURL != nil {
		parsedURL, err := url.Parse(*mount.BucketEndpointURL)
		if err != nil {
			return nil, fmt.Errorf("invalid bucket endpoint URL: %w", err)
		}

		hostname := parsedURL.Hostname()
		if strings.HasSuffix(hostname, "r2.cloudflarestorage.com") {
			mount.bucketType = pb.CloudBucketMount_R2
		} else if strings.HasSuffix(hostname, "storage.googleapis.com") {
			mount.bucketType = pb.CloudBucketMount_GCP
		} else {
			mount.bucketType = pb.CloudBucketMount_S3
			if s.client != nil && s.client.logger != nil {
				s.client.logger.DebugContext(
					ctx,
					"CloudBucketMount received unrecognized bucket endpoint URL. Assuming AWS S3 configuration as fallback.",
					"BucketEndpointURL", *mount.BucketEndpointURL,
				)
			}
		}
	} else {
		mount.bucketType = pb.CloudBucketMount_S3
	}

	if mount.RequesterPays && mount.Secret == nil {
		return nil, fmt.Errorf("credentials required in order to use Requester Pays")
	}

	if mount.KeyPrefix != nil && !strings.HasSuffix(*mount.KeyPrefix, "/") {
		return nil, fmt.Errorf("keyPrefix will be prefixed to all object paths, so it must end in a '/'")
	}

	return mount, nil
}

func (c *CloudBucketMount) toProto(mountPath string) (*pb.CloudBucketMount, error) {
	credentialsSecretID := ""
	if c.Secret != nil {
		credentialsSecretID = c.Secret.SecretID
	}

	return pb.CloudBucketMount_builder{
		BucketName:          c.BucketName,
		MountPath:           mountPath,
		CredentialsSecretId: credentialsSecretID,
		ReadOnly:            c.ReadOnly,
		BucketType:          c.bucketType,
		RequesterPays:       c.RequesterPays,
		BucketEndpointUrl:   c.BucketEndpointURL,
		KeyPrefix:           c.KeyPrefix,
		OidcAuthRoleArn:     c.OidcAuthRoleArn,
	}.Build(), nil
}
