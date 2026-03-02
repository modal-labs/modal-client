package modal

// Function calls and invocations, to be used with Modal Functions.

import (
	"bytes"
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/fxamacker/cbor/v2"
	pickle "github.com/kisielk/og-rek"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// FunctionService provides Function related operations.
type FunctionService interface {
	FromName(ctx context.Context, appName string, name string, params *FunctionFromNameParams) (*Function, error)
}

type functionServiceImpl struct{ client *Client }

// From: modal/_utils/blob_utils.py
const maxObjectSizeBytes int = 2 * 1024 * 1024 // 2 MiB

// From: modal-client/modal/_utils/function_utils.py
const outputsTimeout time.Duration = time.Second * 55

// From: client/modal/_functions.py
const maxSystemRetries = 8

func timeNowSeconds() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}

// FunctionStats represents statistics for a running Function.
type FunctionStats struct {
	Backlog         int
	NumTotalRunners int
}

// FunctionUpdateAutoscalerParams contains options for overriding a Function's autoscaler behavior.
type FunctionUpdateAutoscalerParams struct {
	MinContainers    *uint32
	MaxContainers    *uint32
	BufferContainers *uint32
	ScaledownWindow  *uint32
}

// Function references a deployed Modal Function.
type Function struct {
	FunctionID     string
	handleMetadata *pb.FunctionHandleMetadata

	client *Client
}

// FunctionFromNameParams are options for client.Functions.FromName.
type FunctionFromNameParams struct {
	Environment     string
	CreateIfMissing bool
}

// FromName references a Function from a deployed App by its name.
func (s *functionServiceImpl) FromName(ctx context.Context, appName string, name string, params *FunctionFromNameParams) (*Function, error) {
	if params == nil {
		params = &FunctionFromNameParams{}
	}

	if strings.Contains(name, ".") {
		parts := strings.SplitN(name, ".", 2)
		clsName := parts[0]
		methodName := parts[1]
		return nil, fmt.Errorf("cannot retrieve Cls methods using Functions.FromName(). Use:\n  cls, _ := client.Cls.FromName(ctx, \"%s\", \"%s\", nil)\n  instance, _ := cls.Instance(ctx, nil)\n  m, _ := instance.Method(\"%s\")", appName, clsName, methodName)
	}

	resp, err := s.client.cpClient.FunctionGet(ctx, pb.FunctionGetRequest_builder{
		AppName:         appName,
		ObjectTag:       name,
		EnvironmentName: environmentName(params.Environment, s.client.profile),
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Function '%s/%s' not found", appName, name)}
	}
	if err != nil {
		return nil, err
	}

	handleMetadata := resp.GetHandleMetadata()
	s.client.logger.DebugContext(ctx, "Retrieved Function",
		"function_id", resp.GetFunctionId(),
		"app_name", appName,
		"function_name", name)
	return &Function{FunctionID: resp.GetFunctionId(), handleMetadata: handleMetadata, client: s.client}, nil
}

// pickleSerialize serializes Go data types to the Python pickle format.
// NOTE: This is only used by Queue operations. Function calls use CBOR only.
func pickleSerialize(v any) (bytes.Buffer, error) {
	var inputBuffer bytes.Buffer

	e := pickle.NewEncoder(&inputBuffer)
	err := e.Encode(v)

	if err != nil {
		return bytes.Buffer{}, fmt.Errorf("error pickling data: %w", err)
	}
	return inputBuffer, nil
}

// pickleDeserialize deserializes from Python pickle into Go basic types.
// NOTE: This is only used by Queue operations. Function calls use CBOR only.
func pickleDeserialize(buffer []byte) (any, error) {
	decoder := pickle.NewDecoder(bytes.NewReader(buffer))
	result, err := decoder.Decode()
	if err != nil {
		return nil, fmt.Errorf("error unpickling data: %w", err)
	}
	return result, nil
}

// cborEncoder is configured with time tags enabled so that time.Time values
// are represented as datetime objects in Python. Uses TimeRFC3339Nano to preserve
// nanosecond precision (Python datetime has microsecond precision).
//
// Both options are required:
//   - Time: TimeRFC3339Nano - specifies the format (RFC3339 with nanosecond precision)
//   - TimeTag: EncTagRequired - wraps the time in CBOR tag 0, signaling it's a datetime
//     Without the tag, Python would receive it as a plain string, not a datetime object.
var cborEncoder, _ = cbor.EncOptions{
	Time:    cbor.TimeRFC3339Nano,
	TimeTag: cbor.EncTagRequired,
}.EncMode()

// cborSerialize serializes Go data types to the CBOR format.
// Uses CBOR time tags so that time.Time values are represented as
// datetime objects in Python.
func cborSerialize(v any) ([]byte, error) {
	data, err := cborEncoder.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("error encoding CBOR data: %w", err)
	}
	return data, nil
}

// cborDeserialize deserializes from CBOR into Go basic types.
func cborDeserialize(buffer []byte) (any, error) {
	var result any
	err := cbor.Unmarshal(buffer, &result)
	if err != nil {
		return nil, fmt.Errorf("error decoding CBOR data: %w", err)
	}
	return result, nil
}

// createInput serializes inputs, makes a function call and returns its ID
func (f *Function) createInput(ctx context.Context, args []any, kwargs map[string]any) (*pb.FunctionInput, error) {

	// Check supported input formats and require CBOR
	supportedInputFormats := f.getSupportedInputFormats()
	cborSupported := false
	for _, format := range supportedInputFormats {
		if format == pb.DataFormat_DATA_FORMAT_CBOR {
			cborSupported = true
			break
		}
	}

	// Error if CBOR is not supported
	if !cborSupported {
		return nil, fmt.Errorf("cannot call Modal Function from Go SDK since it was deployed with an incompatible Python SDK version. Redeploy with Modal Python SDK >= 1.2")
	}

	// Use CBOR encoding
	// Ensure args and kwargs are not nil to match expected behavior
	cborArgs := args
	if cborArgs == nil {
		cborArgs = []any{}
	}
	cborKwargs := kwargs
	if cborKwargs == nil {
		cborKwargs = map[string]any{}
	}
	argsBytes, err := cborSerialize([]any{cborArgs, cborKwargs})
	if err != nil {
		return nil, err
	}
	dataFormat := pb.DataFormat_DATA_FORMAT_CBOR
	var argsBlobID *string
	if len(argsBytes) > maxObjectSizeBytes {
		blobID, err := blobUpload(ctx, f.client.cpClient, f.client.logger, argsBytes)
		if err != nil {
			return nil, err
		}
		argsBytes = nil
		argsBlobID = &blobID
	}
	metadata, err := f.getHandleMetadata()
	if err != nil {
		return nil, err
	}
	methodName := metadata.GetUseMethodName() // this is empty if the function is not a cls method
	return pb.FunctionInput_builder{
		Args:       argsBytes,
		ArgsBlobId: argsBlobID,
		DataFormat: dataFormat,
		MethodName: &methodName,
	}.Build(), nil
}

// getHandleMetadata returns the function's handle metadata or an error if not set.
func (f *Function) getHandleMetadata() (*pb.FunctionHandleMetadata, error) {
	if f.handleMetadata == nil {
		return nil, fmt.Errorf("unexpected error: function has not been hydrated")
	}
	return f.handleMetadata, nil
}

// getSupportedInputFormats returns the supported input formats for this function.
// Returns an empty slice if metadata is not available.
func (f *Function) getSupportedInputFormats() []pb.DataFormat {
	metadata, err := f.getHandleMetadata()
	if err != nil {
		// Return empty slice if metadata is not available - this will cause CBOR validation to fail
		return []pb.DataFormat{}
	}
	if len(metadata.GetSupportedInputFormats()) > 0 {
		return metadata.GetSupportedInputFormats()
	}
	return []pb.DataFormat{}
}

// getWebURL returns the web URL for this function, if it's a web endpoint.
// Returns empty string if metadata is not available or if not a web endpoint.
func (f *Function) getWebURL() string {
	metadata, err := f.getHandleMetadata()
	if err != nil {
		return ""
	}
	return metadata.GetWebUrl()
}

func (f *Function) checkNoWebURL(fnName string) error {
	webURL := f.getWebURL()
	if webURL != "" {
		return InvalidError{fmt.Sprintf(
			"A webhook Function cannot be invoked for remote execution with '%s'. Invoke this Function via its web url '%s' instead",
			fnName, webURL,
		)}
	}
	return nil
}

// Remote executes a single input on a remote Function.
func (f *Function) Remote(ctx context.Context, args []any, kwargs map[string]any) (any, error) {
	f.client.logger.DebugContext(ctx, "Executing function call", "function_id", f.FunctionID)
	if err := f.checkNoWebURL("Remote"); err != nil {
		return nil, err
	}
	input, err := f.createInput(ctx, args, kwargs)
	if err != nil {
		return nil, err
	}
	invocation, err := f.createRemoteInvocation(ctx, input)
	if err != nil {
		return nil, err
	}
	// TODO(ryan): Add tests for retries.
	retryCount := uint32(0)
	for {
		output, err := invocation.awaitOutput(ctx, nil)
		if err == nil {
			f.client.logger.DebugContext(ctx, "Function call completed", "function_id", f.FunctionID)
			return output, nil
		}
		if errors.As(err, &InternalFailure{}) && retryCount <= maxSystemRetries {
			f.client.logger.DebugContext(ctx, "Retrying function call due to internal failure",
				"function_id", f.FunctionID,
				"retry_count", retryCount)
			if retryErr := invocation.retry(ctx, retryCount); retryErr != nil {
				return nil, retryErr
			}
			retryCount++
			continue
		}
		return nil, err
	}
}

// createRemoteInvocation creates an Invocation using either the input plane or control plane.
func (f *Function) createRemoteInvocation(ctx context.Context, input *pb.FunctionInput) (invocation, error) {
	metadata, err := f.getHandleMetadata()
	if err != nil {
		return nil, err
	}
	inputPlaneURL := metadata.GetInputPlaneUrl()
	if inputPlaneURL != "" {
		ipClient, err := f.client.ipClient(ctx, inputPlaneURL)
		if err != nil {
			return nil, err
		}
		return createInputPlaneInvocation(ctx, ipClient, f.client.logger, f.FunctionID, input)
	}
	return createControlPlaneInvocation(ctx, f.client.cpClient, f.client.logger, f.FunctionID, input, pb.FunctionCallInvocationType_FUNCTION_CALL_INVOCATION_TYPE_SYNC)
}

// Spawn starts running a single input on a remote Function.
func (f *Function) Spawn(ctx context.Context, args []any, kwargs map[string]any) (*FunctionCall, error) {
	f.client.logger.DebugContext(ctx, "Spawning function call", "function_id", f.FunctionID)
	if err := f.checkNoWebURL("Spawn"); err != nil {
		return nil, err
	}
	input, err := f.createInput(ctx, args, kwargs)
	if err != nil {
		return nil, err
	}
	invocation, err := createControlPlaneInvocation(ctx, f.client.cpClient, f.client.logger, f.FunctionID, input, pb.FunctionCallInvocationType_FUNCTION_CALL_INVOCATION_TYPE_SYNC)
	if err != nil {
		return nil, err
	}
	functionCall := FunctionCall{
		FunctionCallID: invocation.FunctionCallID,
		client:         f.client,
	}
	f.client.logger.DebugContext(ctx, "Function call spawned",
		"function_id", f.FunctionID,
		"function_call_id", invocation.FunctionCallID)
	return &functionCall, nil
}

// GetCurrentStats returns a FunctionStats object with statistics about the Function.
func (f *Function) GetCurrentStats(ctx context.Context) (*FunctionStats, error) {
	resp, err := f.client.cpClient.FunctionGetCurrentStats(ctx, pb.FunctionGetCurrentStatsRequest_builder{
		FunctionId: f.FunctionID,
	}.Build())
	if err != nil {
		return nil, err
	}

	return &FunctionStats{
		Backlog:         int(resp.GetBacklog()),
		NumTotalRunners: int(resp.GetNumTotalTasks()),
	}, nil
}

// UpdateAutoscaler overrides the current autoscaler behavior for this Function.
func (f *Function) UpdateAutoscaler(ctx context.Context, params *FunctionUpdateAutoscalerParams) error {
	if params == nil {
		params = &FunctionUpdateAutoscalerParams{}
	}

	settings := pb.AutoscalerSettings_builder{
		MinContainers:    params.MinContainers,
		MaxContainers:    params.MaxContainers,
		BufferContainers: params.BufferContainers,
		ScaledownWindow:  params.ScaledownWindow,
	}.Build()

	_, err := f.client.cpClient.FunctionUpdateSchedulingParams(ctx, pb.FunctionUpdateSchedulingParamsRequest_builder{
		FunctionId:           f.FunctionID,
		WarmPoolSizeOverride: 0, // Deprecated field, always set to 0
		Settings:             settings,
	}.Build())

	return err
}

// GetWebURL returns the URL of a Function running as a web endpoint.
// Returns empty string if this Function is not a web endpoint.
func (f *Function) GetWebURL() string {
	return f.getWebURL()
}

// blobUpload uploads a blob to storage and returns its ID.
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
		// Skip client-side ETag header validation for now (MD5 checksum).
		return resp.GetBlobId(), nil

	default:
		return "", fmt.Errorf("missing upload URL in BlobCreate response")
	}
}
