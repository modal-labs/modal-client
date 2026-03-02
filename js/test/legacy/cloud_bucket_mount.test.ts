import { CloudBucketMount, Secret } from "modal";
import { CloudBucketMount_BucketType } from "../../proto/modal_proto/api";
import { expect, test } from "vitest";

test("CloudBucketMount constructor with minimal options", () => {
  const mount = new CloudBucketMount("my-bucket");

  expect(mount.bucketName).toBe("my-bucket");
  expect(mount.readOnly).toBe(false);
  expect(mount.requesterPays).toBe(false);
  expect(mount.secret).toBeUndefined();
  expect(mount.bucketEndpointUrl).toBeUndefined();
  expect(mount.keyPrefix).toBeUndefined();
  expect(mount.oidcAuthRoleArn).toBeUndefined();
});

test("CloudBucketMount constructor with all options", () => {
  const mockSecret = { secretId: "sec-123" } as Secret;

  const mount = new CloudBucketMount("my-bucket", {
    secret: mockSecret,
    readOnly: true,
    requesterPays: true,
    bucketEndpointUrl: "https://my-bucket.r2.cloudflarestorage.com",
    keyPrefix: "prefix/",
    oidcAuthRoleArn: "arn:aws:iam::123456789:role/MyRole",
  });

  expect(mount.bucketName).toBe("my-bucket");
  expect(mount.readOnly).toBe(true);
  expect(mount.requesterPays).toBe(true);
  expect(mount.secret).toBe(mockSecret);
  expect(mount.bucketEndpointUrl).toBe(
    "https://my-bucket.r2.cloudflarestorage.com",
  );
  expect(mount.keyPrefix).toBe("prefix/");
  expect(mount.oidcAuthRoleArn).toBe("arn:aws:iam::123456789:role/MyRole");
});

test("CloudBucketMount validation: requesterPays without secret", () => {
  expect(() => {
    new CloudBucketMount("my-bucket", {
      requesterPays: true,
    });
  }).toThrowError("Credentials required in order to use Requester Pays.");
});

test("CloudBucketMount validation: keyPrefix without trailing slash", () => {
  expect(() => {
    new CloudBucketMount("my-bucket", {
      keyPrefix: "prefix",
    });
  }).toThrowError(
    "keyPrefix will be prefixed to all object paths, so it must end in a '/'",
  );
});

test("cloudBucketMount.toProto() with minimal options", () => {
  const mount = new CloudBucketMount("my-bucket");
  const proto = mount.toProto("/mnt/bucket");

  expect(proto.bucketName).toBe("my-bucket");
  expect(proto.mountPath).toBe("/mnt/bucket");
  expect(proto.credentialsSecretId).toBe("");
  expect(proto.readOnly).toBe(false);
  expect(proto.bucketType).toBe(CloudBucketMount_BucketType.S3);
  expect(proto.requesterPays).toBe(false);
  expect(proto.bucketEndpointUrl).toBeUndefined();
  expect(proto.keyPrefix).toBeUndefined();
  expect(proto.oidcAuthRoleArn).toBeUndefined();
});

test("cloudBucketMount.toProto() with all options", () => {
  const mockSecret = { secretId: "sec-123" } as Secret;

  const mount = new CloudBucketMount("my-bucket", {
    secret: mockSecret,
    readOnly: true,
    requesterPays: true,
    bucketEndpointUrl: "https://my-bucket.r2.cloudflarestorage.com",
    keyPrefix: "prefix/",
    oidcAuthRoleArn: "arn:aws:iam::123456789:role/MyRole",
  });

  const proto = mount.toProto("/mnt/bucket");

  expect(proto.bucketName).toBe("my-bucket");
  expect(proto.mountPath).toBe("/mnt/bucket");
  expect(proto.credentialsSecretId).toBe("sec-123");
  expect(proto.readOnly).toBe(true);
  expect(proto.bucketType).toBe(CloudBucketMount_BucketType.R2);
  expect(proto.requesterPays).toBe(true);
  expect(proto.bucketEndpointUrl).toBe(
    "https://my-bucket.r2.cloudflarestorage.com",
  );
  expect(proto.keyPrefix).toBe("prefix/");
  expect(proto.oidcAuthRoleArn).toBe("arn:aws:iam::123456789:role/MyRole");
});
