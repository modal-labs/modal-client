package modal

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/protobuf/proto"
)

type invocation interface {
	awaitOutput(ctx context.Context, timeout *time.Duration) (any, error)
	retry(ctx context.Context, retryCount uint32) error
}

// controlPlaneInvocation implements the invocation interface.
type controlPlaneInvocation struct {
	FunctionCallID  string
	input           *pb.FunctionInput
	functionCallJwt string
	inputJwt        string

	cpClient pb.ModalClientClient
	logger   *slog.Logger
}

// createControlPlaneInvocation executes a function call and returns a new controlPlaneInvocation.
func createControlPlaneInvocation(ctx context.Context, cpClient pb.ModalClientClient, logger *slog.Logger, functionID string, input *pb.FunctionInput, invocationType pb.FunctionCallInvocationType) (*controlPlaneInvocation, error) {
	functionPutInputsItem := pb.FunctionPutInputsItem_builder{
		Idx:   0,
		Input: input,
	}.Build()

	functionMapResponse, err := cpClient.FunctionMap(ctx, pb.FunctionMapRequest_builder{
		FunctionId:                 functionID,
		FunctionCallType:           pb.FunctionCallType_FUNCTION_CALL_TYPE_UNARY,
		FunctionCallInvocationType: invocationType,
		PipelinedInputs:            []*pb.FunctionPutInputsItem{functionPutInputsItem},
	}.Build())
	if err != nil {
		return nil, err
	}

	return &controlPlaneInvocation{
		FunctionCallID:  functionMapResponse.GetFunctionCallId(),
		input:           input,
		functionCallJwt: functionMapResponse.GetFunctionCallJwt(),
		inputJwt:        functionMapResponse.GetPipelinedInputs()[0].GetInputJwt(),
		cpClient:        cpClient,
		logger:          logger,
	}, nil
}

// controlPlaneInvocationFromFunctionCallID creates a controlPlaneInvocation from a function call ID.
func controlPlaneInvocationFromFunctionCallID(cpClient pb.ModalClientClient, logger *slog.Logger, functionCallID string) *controlPlaneInvocation {
	return &controlPlaneInvocation{FunctionCallID: functionCallID, cpClient: cpClient, logger: logger}
}

func (c *controlPlaneInvocation) awaitOutput(ctx context.Context, timeout *time.Duration) (any, error) {
	return pollFunctionOutput(ctx, c.cpClient, c.logger, c.getOutput, timeout)
}

func (c *controlPlaneInvocation) retry(ctx context.Context, retryCount uint32) error {
	if c.input == nil {
		return fmt.Errorf("cannot retry Function invocation - input missing")
	}
	retryItem := pb.FunctionRetryInputsItem_builder{
		InputJwt:   c.inputJwt,
		Input:      c.input,
		RetryCount: retryCount,
	}.Build()
	functionRetryResponse, err := c.cpClient.FunctionRetryInputs(ctx, pb.FunctionRetryInputsRequest_builder{
		FunctionCallJwt: c.functionCallJwt,
		Inputs:          []*pb.FunctionRetryInputsItem{retryItem},
	}.Build())
	if err != nil {
		return err
	}
	c.inputJwt = functionRetryResponse.GetInputJwts()[0]
	return nil
}

// getOutput fetches the output for the current function call.
func (c *controlPlaneInvocation) getOutput(ctx context.Context, timeout time.Duration) (*pb.FunctionGetOutputsItem, error) {
	response, err := c.cpClient.FunctionGetOutputs(ctx, pb.FunctionGetOutputsRequest_builder{
		FunctionCallId: c.FunctionCallID,
		MaxValues:      1,
		Timeout:        float32(timeout.Seconds()),
		LastEntryId:    "0-0",
		ClearOnSuccess: true,
		RequestedAt:    timeNowSeconds(),
	}.Build())
	if err != nil {
		return nil, fmt.Errorf("FunctionGetOutputs failed: %w", err)
	}
	outputs := response.GetOutputs()
	if len(outputs) > 0 {
		return outputs[0], nil
	}
	return nil, nil
}

// inputPlaneInvocation implements the Invocation interface for the input plane.
type inputPlaneInvocation struct {
	functionID   string
	input        *pb.FunctionPutInputsItem
	attemptToken string

	ipClient pb.ModalClientClient
	logger   *slog.Logger
}

// createInputPlaneInvocation creates a new InputPlaneInvocation by starting an attempt.
func createInputPlaneInvocation(ctx context.Context, ipClient pb.ModalClientClient, logger *slog.Logger, functionID string, input *pb.FunctionInput) (*inputPlaneInvocation, error) {
	functionPutInputsItem := pb.FunctionPutInputsItem_builder{
		Idx:   0,
		Input: input,
	}.Build()
	attemptStartResp, err := ipClient.AttemptStart(ctx, pb.AttemptStartRequest_builder{
		FunctionId: functionID,
		Input:      functionPutInputsItem,
	}.Build())
	if err != nil {
		return nil, err
	}
	return &inputPlaneInvocation{
		functionID:   functionID,
		input:        functionPutInputsItem,
		attemptToken: attemptStartResp.GetAttemptToken(),
		ipClient:     ipClient,
		logger:       logger,
	}, nil
}

// awaitOutput waits for the output with an optional timeout.
func (i *inputPlaneInvocation) awaitOutput(ctx context.Context, timeout *time.Duration) (any, error) {
	return pollFunctionOutput(ctx, i.ipClient, i.logger, i.getOutput, timeout)
}

// getOutput fetches the output for the current attempt.
func (i *inputPlaneInvocation) getOutput(ctx context.Context, timeout time.Duration) (*pb.FunctionGetOutputsItem, error) {
	resp, err := i.ipClient.AttemptAwait(ctx, pb.AttemptAwaitRequest_builder{
		AttemptToken: i.attemptToken,
		RequestedAt:  timeNowSeconds(),
		TimeoutSecs:  float32(timeout.Seconds()),
	}.Build())
	if err != nil {
		return nil, fmt.Errorf("AttemptAwait failed: %w", err)
	}
	return resp.GetOutput(), nil
}

// retry retries the invocation.
func (i *inputPlaneInvocation) retry(ctx context.Context, retryCount uint32) error {
	// We ignore retryCount - it is used only by controlPlaneInvocation.
	resp, err := i.ipClient.AttemptRetry(ctx, pb.AttemptRetryRequest_builder{
		FunctionId:   i.functionID,
		Input:        i.input,
		AttemptToken: i.attemptToken,
	}.Build())
	if err != nil {
		return err
	}
	i.attemptToken = resp.GetAttemptToken()
	return nil
}

// getOutput is a function type that takes a timeout and returns a FunctionGetOutputsItem or nil, and an error.
// Used by `pollForOutputs` to fetch from either the control plane or the input plane, depending on the implementation.
type getOutput func(ctx context.Context, timeout time.Duration) (*pb.FunctionGetOutputsItem, error)

// pollFunctionOutput repeatedly tries to fetch an output using the provided `getOutput` function, and the specified
// timeout value. We use a timeout value of 55 seconds if the caller does not specify a timeout value, or if the
// specified timeout value is greater than 55 seconds.
func pollFunctionOutput(ctx context.Context, client pb.ModalClientClient, logger *slog.Logger, getOutput getOutput, timeout *time.Duration) (any, error) {
	startTime := time.Now()
	pollTimeout := outputsTimeout
	if timeout != nil {
		// Refresh backend call once per outputsTimeout.
		pollTimeout = min(*timeout, outputsTimeout)
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		output, err := getOutput(ctx, pollTimeout)
		if err != nil {
			return nil, err
		}
		// Output serialization may fail if any of the output items can't be deserialized
		// into a supported Go type. Users are expected to serialize outputs correctly.
		if output != nil {
			return processResult(ctx, client, logger, output.GetResult(), output.GetDataFormat())
		}

		if timeout != nil {
			remainingTime := *timeout - time.Since(startTime)
			if remainingTime <= 0 {
				message := fmt.Sprintf("Timeout exceeded: %.1fs", timeout.Seconds())
				return nil, FunctionTimeoutError{message}
			}
			pollTimeout = min(outputsTimeout, remainingTime)
		}
	}
}

// processResult processes the result from an invocation.
func processResult(ctx context.Context, client pb.ModalClientClient, logger *slog.Logger, result *pb.GenericResult, dataFormat pb.DataFormat) (any, error) {
	if result == nil {
		return nil, RemoteError{"Received null result from invocation"}
	}

	var data []byte
	var err error
	switch result.WhichDataOneof() {
	case pb.GenericResult_Data_case:
		data = result.GetData()
	case pb.GenericResult_DataBlobId_case:
		data, err = blobDownload(ctx, client, logger, result.GetDataBlobId())
		if err != nil {
			return nil, err
		}
	case pb.GenericResult_DataOneof_not_set_case:
		data = nil
	}

	switch result.GetStatus() {
	case pb.GenericResult_GENERIC_STATUS_TIMEOUT:
		return nil, FunctionTimeoutError{result.GetException()}
	case pb.GenericResult_GENERIC_STATUS_INTERNAL_FAILURE:
		return nil, InternalFailure{result.GetException()}
	case pb.GenericResult_GENERIC_STATUS_SUCCESS:
		// Proceed to the block below this switch statement.
	default:
		// In this case, `result.GetData()` may have a pickled user code exception with traceback
		// from Python. We ignore this and only take the string representation.
		exception := result.GetException()
		return nil, RemoteError{exception}
	}

	return deserializeDataFormat(data, dataFormat)
}

// blobDownload downloads a blob by its ID.
func blobDownload(ctx context.Context, client pb.ModalClientClient, logger *slog.Logger, blobID string) ([]byte, error) {
	resp, err := client.BlobGet(ctx, pb.BlobGetRequest_builder{
		BlobId: blobID,
	}.Build())
	if err != nil {
		return nil, err
	}
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
	buf, err := io.ReadAll(s3resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read blob data: %w", err)
	}
	return buf, nil
}

func deserializeDataFormat(data []byte, dataFormat pb.DataFormat) (any, error) {
	switch dataFormat {
	case pb.DataFormat_DATA_FORMAT_CBOR:
		return cborDeserialize(data)
	case pb.DataFormat_DATA_FORMAT_PICKLE:
		return nil, fmt.Errorf("PICKLE output format is not supported - remote function must return CBOR format")
	case pb.DataFormat_DATA_FORMAT_ASGI:
		return nil, fmt.Errorf("ASGI data format is not supported in Go")
	case pb.DataFormat_DATA_FORMAT_GENERATOR_DONE:
		var done pb.GeneratorDone
		if err := proto.Unmarshal(data, &done); err != nil {
			return nil, fmt.Errorf("failed to unmarshal GeneratorDone: %w", err)
		}
		return &done, nil
	default:
		return nil, fmt.Errorf("unsupported data format: %s", dataFormat.String())
	}
}
