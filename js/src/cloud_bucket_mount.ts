import {
  CloudBucketMount_BucketType,
  CloudBucketMount as CloudBucketMountProto,
} from "../proto/modal_proto/api";
import { ModalClient } from "./client";
import { Secret } from "./secret";

export class CloudBucketMountService {
  readonly #client: ModalClient;

  constructor(client: ModalClient) {
    this.#client = client;
  }

  create(
    bucketName: string,
    params: {
      secret?: Secret;
      readOnly?: boolean;
      requesterPays?: boolean;
      bucketEndpointUrl?: string;
      keyPrefix?: string;
      oidcAuthRoleArn?: string;
    } = {},
  ): CloudBucketMount {
    let bucketType = CloudBucketMount_BucketType.S3;
    if (params.bucketEndpointUrl) {
      const url = new URL(params.bucketEndpointUrl);
      if (url.hostname.endsWith("r2.cloudflarestorage.com")) {
        bucketType = CloudBucketMount_BucketType.R2;
      } else if (url.hostname.endsWith("storage.googleapis.com")) {
        bucketType = CloudBucketMount_BucketType.GCP;
      } else {
        bucketType = CloudBucketMount_BucketType.S3;
        this.#client.logger.debug(
          "CloudBucketMount received unrecognized bucket endpoint URL. " +
            "Assuming AWS S3 configuration as fallback.",
          "bucketEndpointUrl",
          params.bucketEndpointUrl,
        );
      }
    }

    if (params.requesterPays && !params.secret) {
      throw new Error("Credentials required in order to use Requester Pays.");
    }

    if (params.keyPrefix && !params.keyPrefix.endsWith("/")) {
      throw new Error(
        "keyPrefix will be prefixed to all object paths, so it must end in a '/'",
      );
    }

    return new CloudBucketMount(
      bucketName,
      params.secret,
      params.readOnly ?? false,
      params.requesterPays ?? false,
      params.bucketEndpointUrl,
      params.keyPrefix,
      params.oidcAuthRoleArn,
      bucketType,
    );
  }
}

/** Cloud Bucket Mounts provide access to cloud storage buckets within Modal Functions. */
export class CloudBucketMount {
  readonly bucketName!: string;
  readonly secret?: Secret;
  readonly readOnly!: boolean;
  readonly requesterPays!: boolean;
  readonly bucketEndpointUrl?: string;
  readonly keyPrefix?: string;
  readonly oidcAuthRoleArn?: string;
  readonly #bucketType!: CloudBucketMount_BucketType;

  /** @ignore */
  constructor(
    bucketName: string,
    secret: Secret | undefined,
    readOnly: boolean,
    requesterPays: boolean,
    bucketEndpointUrl: string | undefined,
    keyPrefix: string | undefined,
    oidcAuthRoleArn: string | undefined,
    bucketType: CloudBucketMount_BucketType,
  ) {
    this.bucketName = bucketName;
    this.secret = secret;
    this.readOnly = readOnly;
    this.requesterPays = requesterPays;
    this.bucketEndpointUrl = bucketEndpointUrl;
    this.keyPrefix = keyPrefix;
    this.oidcAuthRoleArn = oidcAuthRoleArn;
    this.#bucketType = bucketType;
  }

  /** @ignore */
  toProto(mountPath: string): CloudBucketMountProto {
    return CloudBucketMountProto.create({
      bucketName: this.bucketName,
      mountPath,
      credentialsSecretId: this.secret?.secretId ?? "",
      readOnly: this.readOnly,
      bucketType: this.#bucketType,
      requesterPays: this.requesterPays,
      bucketEndpointUrl: this.bucketEndpointUrl,
      keyPrefix: this.keyPrefix,
      oidcAuthRoleArn: this.oidcAuthRoleArn,
    });
  }
}
