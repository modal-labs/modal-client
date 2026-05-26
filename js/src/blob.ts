import { createHash } from "node:crypto";

import { ModalGrpcClient } from "./client";

const blobUploadRetryAttempts = 3;
const blobUploadRetryDelayMs = 300;
const blobDownloadRetryAttempts = 5;
const blobDownloadRetryDelayMs = 100;

async function retryHttpRequest<T>(
  fn: () => Promise<T>,
  attempts: number,
  baseDelayMs: number,
): Promise<T> {
  let delay = baseDelayMs;
  let lastErr: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, delay));
        delay *= 2;
      }
    }
  }
  throw lastErr;
}

export async function blobUpload(
  cpClient: ModalGrpcClient,
  data: Uint8Array,
): Promise<string> {
  const contentMd5 = createHash("md5").update(data).digest("base64");
  const contentSha256 = createHash("sha256").update(data).digest("base64");
  const resp = await cpClient.blobCreate({
    contentMd5,
    contentSha256Base64: contentSha256,
    contentLength: data.length,
  });
  if (resp.multipart) {
    throw new Error(
      "Function input size exceeds multipart upload threshold, unsupported by this SDK version",
    );
  } else if (resp.uploadUrl) {
    await retryHttpRequest(
      async () => {
        const uploadResp = await fetch(resp.uploadUrl!, {
          method: "PUT",
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-MD5": contentMd5,
          },
          body: data,
        });
        if (uploadResp.status < 200 || uploadResp.status >= 300) {
          throw new Error(`Failed blob upload: ${uploadResp.statusText}`);
        }
      },
      blobUploadRetryAttempts,
      blobUploadRetryDelayMs,
    );
    // Skip client-side ETag header validation for now (MD5 checksum).
    return resp.blobId;
  } else {
    throw new Error("Missing upload URL in BlobCreate response");
  }
}

export async function blobDownload(
  cpClient: ModalGrpcClient,
  blobId: string,
): Promise<Uint8Array> {
  const resp = await cpClient.blobGet({ blobId });
  return retryHttpRequest(
    async () => {
      const s3resp = await fetch(resp.downloadUrl);
      if (!s3resp.ok) {
        throw new Error(`Failed to download blob: ${s3resp.statusText}`);
      }
      const buf = await s3resp.arrayBuffer();
      return new Uint8Array(buf);
    },
    blobDownloadRetryAttempts,
    blobDownloadRetryDelayMs,
  );
}

// Exported for testing only.
export { retryHttpRequest as _retryHttpRequest };
