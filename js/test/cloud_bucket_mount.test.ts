import { tc } from "../test-support/test-client";
import { Secret } from "modal";
import { CloudBucketMount_BucketType } from "../proto/modal_proto/api";
import { expect, test } from "vitest";

test("CloudBucketMountService.create() with minimal options", () => {
  const mount = tc.cloudBucketMounts.create("my-bucket");

  expect(mount.bucketName).toBe("my-bucket");
  expect(mount.readOnly).toBe(false);
  expect(mount.requesterPays).toBe(false);
  expect(mount.secret).toBeUndefined();
  expect(mount.bucketEndpointUrl).toBeUndefined();
  expect(mount.keyPrefix).toBeUndefined();
  expect(mount.oidcAuthRoleArn).toBeUndefined();

  expect(mount.toProto("/").bucketType).toEqual(CloudBucketMount_BucketType.S3);
});

test("CloudBucketMountService.create() with all options", () => {
  const mockSecret = { secretId: "sec-123" } as Secret;

  const mount = tc.cloudBucketMounts.create("my-bucket", {
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

  expect(mount.toProto("/").bucketType).toEqual(CloudBucketMount_BucketType.R2);
});

test("CloudBucketMountService.create() bucket type detection from endpoint URLs", () => {
  expect(
    tc.cloudBucketMounts
      .create("my-bucket", {
        bucketEndpointUrl: "",
      })
      .toProto("/").bucketType,
  ).toEqual(CloudBucketMount_BucketType.S3);

  expect(
    tc.cloudBucketMounts
      .create("my-bucket", {
        bucketEndpointUrl: "https://my-bucket.r2.cloudflarestorage.com",
      })
      .toProto("/").bucketType,
  ).toEqual(CloudBucketMount_BucketType.R2);

  expect(
    tc.cloudBucketMounts
      .create("my-bucket", {
        bucketEndpointUrl: "https://storage.googleapis.com/my-bucket",
      })
      .toProto("/").bucketType,
  ).toEqual(CloudBucketMount_BucketType.GCP);

  expect(
    tc.cloudBucketMounts
      .create("my-bucket", {
        bucketEndpointUrl: "https://unknown-endpoint.com/my-bucket",
      })
      .toProto("/").bucketType,
  ).toEqual(CloudBucketMount_BucketType.S3);

  expect(() => {
    tc.cloudBucketMounts.create("my-bucket", {
      bucketEndpointUrl: "://invalid-url",
    });
  }).toThrowError("Invalid URL");
});

test("CloudBucketMountService.create() validation: requesterPays without secret", () => {
  expect(() => {
    tc.cloudBucketMounts.create("my-bucket", {
      requesterPays: true,
    });
  }).toThrowError("Credentials required in order to use Requester Pays.");
});

test("CloudBucketMountService.create() validation: keyPrefix without trailing slash", () => {
  expect(() => {
    tc.cloudBucketMounts.create("my-bucket", {
      keyPrefix: "prefix",
    });
  }).toThrowError(
    "keyPrefix will be prefixed to all object paths, so it must end in a '/'",
  );
});

test("cloudBucketMount.toProto() with minimal options", () => {
  const mount = tc.cloudBucketMounts.create("my-bucket");
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

  const mount = tc.cloudBucketMounts.create("my-bucket", {
    secret: mockSecret,
    readOnly: true,
    requesterPays: true,
    bucketEndpointUrl: "https://storage.googleapis.com/my-bucket",
    keyPrefix: "prefix/",
    oidcAuthRoleArn: "arn:aws:iam::123456789:role/MyRole",
  });

  const proto = mount.toProto("/mnt/bucket");

  expect(proto.bucketName).toBe("my-bucket");
  expect(proto.mountPath).toBe("/mnt/bucket");
  expect(proto.credentialsSecretId).toBe("sec-123");
  expect(proto.readOnly).toBe(true);
  expect(proto.bucketType).toBe(CloudBucketMount_BucketType.GCP);
  expect(proto.requesterPays).toBe(true);
  expect(proto.bucketEndpointUrl).toBe(
    "https://storage.googleapis.com/my-bucket",
  );
  expect(proto.keyPrefix).toBe("prefix/");
  expect(proto.oidcAuthRoleArn).toBe("arn:aws:iam::123456789:role/MyRole");
});
