package modal

import (
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

func newTestMount(bucketName string, params *CloudBucketMountParams) (*CloudBucketMount, error) {
	return (&cloudBucketMountServiceImpl{}).New(bucketName, params)
}

func TestNewCloudBucketMount_MinimalOptions(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mount, err := newTestMount("my-bucket", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mount.BucketName).Should(gomega.Equal("my-bucket"))
	g.Expect(mount.ReadOnly).Should(gomega.BeFalse())
	g.Expect(mount.RequesterPays).Should(gomega.BeFalse())
	g.Expect(mount.Secret).Should(gomega.BeNil())
	g.Expect(mount.BucketEndpointURL).Should(gomega.BeNil())
	g.Expect(mount.KeyPrefix).Should(gomega.BeNil())
	g.Expect(mount.OidcAuthRoleArn).Should(gomega.BeNil())
	g.Expect(mount.bucketType).Should(gomega.Equal(pb.CloudBucketMount_S3))
}

func TestNewCloudBucketMount_AllOptions(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockSecret := &Secret{SecretID: "sec-123"}
	endpointURL := "https://my-bucket.r2.cloudflarestorage.com"
	keyPrefix := "prefix/"
	oidcRole := "arn:aws:iam::123456789:role/MyRole"

	mount, err := newTestMount("my-bucket", &CloudBucketMountParams{
		Secret:            mockSecret,
		ReadOnly:          true,
		RequesterPays:     true,
		BucketEndpointURL: &endpointURL,
		KeyPrefix:         &keyPrefix,
		OidcAuthRoleArn:   &oidcRole,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(mount.BucketName).Should(gomega.Equal("my-bucket"))
	g.Expect(mount.ReadOnly).Should(gomega.BeTrue())
	g.Expect(mount.RequesterPays).Should(gomega.BeTrue())
	g.Expect(mount.Secret).Should(gomega.Equal(mockSecret))
	g.Expect(mount.BucketEndpointURL).ShouldNot(gomega.BeNil())
	g.Expect(*mount.BucketEndpointURL).Should(gomega.Equal(endpointURL))
	g.Expect(mount.KeyPrefix).ShouldNot(gomega.BeNil())
	g.Expect(*mount.KeyPrefix).Should(gomega.Equal(keyPrefix))
	g.Expect(mount.OidcAuthRoleArn).ShouldNot(gomega.BeNil())
	g.Expect(*mount.OidcAuthRoleArn).Should(gomega.Equal(oidcRole))
	g.Expect(mount.bucketType).Should(gomega.Equal(pb.CloudBucketMount_R2))
}

func TestBucketTypeDetection(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name         string
		endpointURL  string
		expectedType pb.CloudBucketMount_BucketType
	}{
		{
			name:         "Empty defaults to S3",
			endpointURL:  "",
			expectedType: pb.CloudBucketMount_S3,
		},
		{
			name:         "R2",
			endpointURL:  "https://my-bucket.r2.cloudflarestorage.com",
			expectedType: pb.CloudBucketMount_R2,
		},
		{
			name:         "GCP",
			endpointURL:  "https://storage.googleapis.com/my-bucket",
			expectedType: pb.CloudBucketMount_GCP,
		},
		{
			name:         "Unknown defaults to S3",
			endpointURL:  "https://unknown-endpoint.com/my-bucket",
			expectedType: pb.CloudBucketMount_S3,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)

			params := &CloudBucketMountParams{}
			if tc.endpointURL != "" {
				params.BucketEndpointURL = &tc.endpointURL
			}

			mount, err := newTestMount("my-bucket", params)
			g.Expect(err).ShouldNot(gomega.HaveOccurred())
			g.Expect(mount.bucketType).Should(gomega.Equal(tc.expectedType))
		})
	}
}

func TestNewCloudBucketMount_InvalidURL(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	invalidURL := "://invalid-url"
	_, err := newTestMount("my-bucket", &CloudBucketMountParams{
		BucketEndpointURL: &invalidURL,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("invalid bucket endpoint URL"))
}

func TestNewCloudBucketMount_ValidationRequesterPaysWithoutSecret(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := newTestMount("my-bucket", &CloudBucketMountParams{
		RequesterPays: true,
	})

	g.Expect(err).Should(gomega.MatchError("credentials required in order to use Requester Pays"))
}

func TestNewCloudBucketMount_ValidationKeyPrefixWithoutTrailingSlash(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	keyPrefix := "prefix"
	_, err := newTestMount("my-bucket", &CloudBucketMountParams{
		KeyPrefix: &keyPrefix,
	})

	g.Expect(err).Should(gomega.MatchError("keyPrefix will be prefixed to all object paths, so it must end in a '/'"))
}

func TestCloudBucketMount_ToProtoMinimalOptions(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mount, err := newTestMount("my-bucket", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	proto, err := mount.toProto("/mnt/bucket")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(proto.GetBucketName()).Should(gomega.Equal("my-bucket"))
	g.Expect(proto.GetMountPath()).Should(gomega.Equal("/mnt/bucket"))
	g.Expect(proto.GetCredentialsSecretId()).Should(gomega.BeEmpty())
	g.Expect(proto.GetReadOnly()).Should(gomega.BeFalse())
	g.Expect(proto.GetBucketType()).Should(gomega.Equal(pb.CloudBucketMount_S3))
	g.Expect(proto.GetRequesterPays()).Should(gomega.BeFalse())
	g.Expect(proto.GetBucketEndpointUrl()).Should(gomega.BeEmpty())
	g.Expect(proto.GetKeyPrefix()).Should(gomega.BeEmpty())
	g.Expect(proto.GetOidcAuthRoleArn()).Should(gomega.BeEmpty())
}

func TestCloudBucketMount_ToProtoAllOptions(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	mockSecret := &Secret{SecretID: "sec-123"}
	endpointURL := "https://my-bucket.r2.cloudflarestorage.com"
	keyPrefix := "prefix/"
	oidcRole := "arn:aws:iam::123456789:role/MyRole"

	mount, err := newTestMount("my-bucket", &CloudBucketMountParams{
		Secret:            mockSecret,
		ReadOnly:          true,
		RequesterPays:     true,
		BucketEndpointURL: &endpointURL,
		KeyPrefix:         &keyPrefix,
		OidcAuthRoleArn:   &oidcRole,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	proto, err := mount.toProto("/mnt/bucket")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(proto.GetBucketName()).Should(gomega.Equal("my-bucket"))
	g.Expect(proto.GetMountPath()).Should(gomega.Equal("/mnt/bucket"))
	g.Expect(proto.GetCredentialsSecretId()).Should(gomega.Equal("sec-123"))
	g.Expect(proto.GetReadOnly()).Should(gomega.BeTrue())
	g.Expect(proto.GetBucketType()).Should(gomega.Equal(pb.CloudBucketMount_R2))
	g.Expect(proto.GetRequesterPays()).Should(gomega.BeTrue())
	g.Expect(proto.GetBucketEndpointUrl()).Should(gomega.Equal(endpointURL))
	g.Expect(proto.GetKeyPrefix()).Should(gomega.Equal(keyPrefix))
	g.Expect(proto.GetOidcAuthRoleArn()).Should(gomega.Equal(oidcRole))
}
