package modal

import (
	"bytes"
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
)

const (
	blobUploadRetryAttempts   = 3
	blobUploadRetryDelay      = 300 * time.Millisecond
	blobDownloadRetryAttempts = 5
	blobDownloadRetryDelay    = 100 * time.Millisecond
)

// retryHTTPRequest retries an HTTP operation with exponential backoff.
func retryHTTPRequest[T any](ctx context.Context, logger *slog.Logger, operation string, attempts int, baseDelay time.Duration, fn func() (T, error)) (T, error) {
	delay := baseDelay
	var zero T
	var lastErr error
	for attempt := range attempts {
		result, err := fn()
		if err == nil {
			return result, nil
		}
		lastErr = err
		if attempt < attempts-1 {
			logger.DebugContext(ctx, fmt.Sprintf("%s failed (attempt %d/%d), retrying",
				operation, attempt+1, attempts), "error", lastErr.Error())
			select {
			case <-ctx.Done():
				return zero, ctx.Err()
			case <-time.After(delay):
			}
			delay *= 2
		}
	}
	return zero, lastErr
}

func blobUpload(ctx context.Context, client pb.ModalClientClient, logger *slog.Logger, data []byte) (string, error) {
	md5sum := md5.Sum(data)
	sha256sum := sha256.Sum256(data)
	contentMd5 := base64.StdEncoding.EncodeToString(md5sum[:])
	contentSha256 := base64.StdEncoding.EncodeToString(sha256sum[:])

	resp, err := client.BlobCreate(ctx, pb.BlobCreateRequest_builder{
		ContentMd5:          contentMd5,
		ContentSha256Base64: contentSha256,
		ContentLength:       int64(len(data)),
	}.Build())
	if err != nil {
		return "", fmt.Errorf("failed to create blob: %w", err)
	}

	switch resp.WhichUploadTypeOneof() {
	case pb.BlobCreateResponse_Multipart_case:
		return "", fmt.Errorf("Function input size exceeds multipart upload threshold, unsupported by this SDK version")

	case pb.BlobCreateResponse_UploadUrl_case:
		return retryHTTPRequest(ctx, logger, "blob upload", blobUploadRetryAttempts, blobUploadRetryDelay, func() (string, error) {
			req, err := http.NewRequestWithContext(ctx, "PUT", resp.GetUploadUrl(), bytes.NewReader(data))
			if err != nil {
				return "", fmt.Errorf("failed to create upload request: %w", err)
			}
			req.Header.Set("Content-Type", "application/octet-stream")
			req.Header.Set("Content-MD5", contentMd5)
			uploadResp, err := http.DefaultClient.Do(req)
			if err != nil {
				return "", fmt.Errorf("failed to upload blob: %w", err)
			}
			defer func() {
				if err := uploadResp.Body.Close(); err != nil {
					logger.DebugContext(ctx, "failed to close upload response body", "error", err.Error())
				}
			}()
			if uploadResp.StatusCode < 200 || uploadResp.StatusCode >= 300 {
				return "", fmt.Errorf("failed blob upload: %s", uploadResp.Status)
			}
			return resp.GetBlobId(), nil
		})

	default:
		return "", fmt.Errorf("missing upload URL in BlobCreate response")
	}
}

// blobDownload downloads a blob by its ID.
func blobDownload(ctx context.Context, client pb.ModalClientClient, logger *slog.Logger, blobID string) ([]byte, error) {
	resp, err := client.BlobGet(ctx, pb.BlobGetRequest_builder{
		BlobId: blobID,
	}.Build())
	if err != nil {
		return nil, err
	}
	return retryHTTPRequest(ctx, logger, "blob download", blobDownloadRetryAttempts, blobDownloadRetryDelay, func() ([]byte, error) {
		req, err := http.NewRequestWithContext(ctx, "GET", resp.GetDownloadUrl(), nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create download request: %w", err)
		}
		s3resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to download blob: %w", err)
		}
		defer func() {
			if err := s3resp.Body.Close(); err != nil {
				logger.DebugContext(ctx, "failed to close download response body", "error", err.Error())
			}
		}()
		if s3resp.StatusCode < 200 || s3resp.StatusCode >= 300 {
			return nil, fmt.Errorf("failed blob download: %s", s3resp.Status)
		}
		buf, err := io.ReadAll(s3resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read blob data: %w", err)
		}
		return buf, nil
	})
}
