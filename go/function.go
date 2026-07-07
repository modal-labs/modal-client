package modal

// Function calls and invocations, to be used with Modal Functions.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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

	client  *Client
	options *functionOptions
}

// FunctionFromNameParams are options for client.Functions.FromName.
type FunctionFromNameParams struct {
	Environment string
	// Version looks up a version-pinned Function deployed at this App version.
	Version int
}

func hasOptions(o *functionOptions) bool {
	return o != nil && *o != (functionOptions{})
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
		EnvironmentName: firstNonEmpty(params.Environment, s.client.profile.Environment),
		AppVersion:      int32(params.Version),
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

	return &Function{
		FunctionID:     resp.GetFunctionId(),
		handleMetadata: handleMetadata,
		client:         s.client,
		options:        &functionOptions{},
	}, nil
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

// getWebURL returns the web URL for this function, if it's a Web Function.
// Returns empty string if metadata is not available or if not a Web Function.
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

// FunctionWithOptionsParams represents runtime options for a Modal Function.
type FunctionWithOptionsParams struct {
	CPU              *float64
	CPULimit         *float64
	MemoryMiB        *int
	MemoryLimitMiB   *int
	GPU              *string
	Env              map[string]string
	Secrets          []*Secret
	Volumes          map[string]*Volume
	Retries          *Retries
	MaxContainers    *int
	BufferContainers *int
	ScaledownWindow  *time.Duration
	Timeout          *time.Duration
	RoutingRegion    *string
}

// FunctionWithConcurrencyParams represents concurrency configuration for a Modal Function.
type FunctionWithConcurrencyParams struct {
	MaxInputs    int
	TargetInputs *int
}

// FunctionWithBatchingParams represents batching configuration for a Modal Function.
type FunctionWithBatchingParams struct {
	MaxBatchSize int
	Wait         time.Duration
}

type functionOptions struct {
	cpu                    *float64
	cpuLimit               *float64
	memoryMiB              *int
	memoryLimitMiB         *int
	gpu                    *string
	env                    *map[string]string
	secrets                *[]*Secret
	volumes                *map[string]*Volume
	retries                *Retries
	maxContainers          *int
	bufferContainers       *int
	scaledownWindow        *time.Duration
	timeout                *time.Duration
	maxConcurrentInputs    *int
	targetConcurrentInputs *int
	batchMaxSize           *int
	batchWait              *time.Duration
	routingRegion          *string
}

func mergeFunctionOptions(base, new *functionOptions) *functionOptions {
	if base == nil {
		return new
	}
	if new == nil {
		return base
	}

	merged := &functionOptions{
		cpu:                    base.cpu,
		cpuLimit:               base.cpuLimit,
		memoryMiB:              base.memoryMiB,
		memoryLimitMiB:         base.memoryLimitMiB,
		gpu:                    base.gpu,
		env:                    base.env,
		secrets:                base.secrets,
		volumes:                base.volumes,
		retries:                base.retries,
		maxContainers:          base.maxContainers,
		bufferContainers:       base.bufferContainers,
		scaledownWindow:        base.scaledownWindow,
		timeout:                base.timeout,
		maxConcurrentInputs:    base.maxConcurrentInputs,
		targetConcurrentInputs: base.targetConcurrentInputs,
		batchMaxSize:           base.batchMaxSize,
		batchWait:              base.batchWait,
		routingRegion:          base.routingRegion,
	}

	if new.cpu != nil {
		merged.cpu = new.cpu
	}
	if new.cpuLimit != nil {
		merged.cpuLimit = new.cpuLimit
	}
	if new.memoryMiB != nil {
		merged.memoryMiB = new.memoryMiB
	}
	if new.memoryLimitMiB != nil {
		merged.memoryLimitMiB = new.memoryLimitMiB
	}
	if new.gpu != nil {
		merged.gpu = new.gpu
	}
	if new.env != nil {
		merged.env = new.env
	}
	if new.secrets != nil {
		merged.secrets = new.secrets
	}
	if new.volumes != nil {
		merged.volumes = new.volumes
	}
	if new.retries != nil {
		merged.retries = new.retries
	}
	if new.maxContainers != nil {
		merged.maxContainers = new.maxContainers
	}
	if new.bufferContainers != nil {
		merged.bufferContainers = new.bufferContainers
	}
	if new.scaledownWindow != nil {
		merged.scaledownWindow = new.scaledownWindow
	}
	if new.timeout != nil {
		merged.timeout = new.timeout
	}
	if new.maxConcurrentInputs != nil {
		merged.maxConcurrentInputs = new.maxConcurrentInputs
	}
	if new.targetConcurrentInputs != nil {
		merged.targetConcurrentInputs = new.targetConcurrentInputs
	}
	if new.batchMaxSize != nil {
		merged.batchMaxSize = new.batchMaxSize
	}
	if new.batchWait != nil {
		merged.batchWait = new.batchWait
	}
	if new.routingRegion != nil {
		merged.routingRegion = new.routingRegion
	}

	return merged
}

func buildFunctionOptionsProto(options *functionOptions) (*pb.FunctionOptions, error) {
	if !hasOptions(options) {
		return nil, nil
	}

	builder := pb.FunctionOptions_builder{}

	if options.cpu != nil || options.cpuLimit != nil || options.memoryMiB != nil || options.memoryLimitMiB != nil || options.gpu != nil {
		resBuilder := pb.Resources_builder{}

		if options.cpu == nil && options.cpuLimit != nil {
			return nil, fmt.Errorf("must also specify non-zero CPU request when CPULimit is specified")
		}
		if options.cpu != nil {
			if *options.cpu <= 0 {
				return nil, fmt.Errorf("the CPU request (%f) must be a positive number", *options.cpu)
			}
			resBuilder.MilliCpu = uint32(*options.cpu * 1000)
			if options.cpuLimit != nil {
				if *options.cpuLimit < *options.cpu {
					return nil, fmt.Errorf("the CPU request (%f) cannot be higher than CPULimit (%f)", *options.cpu, *options.cpuLimit)
				}
				resBuilder.MilliCpuMax = uint32(*options.cpuLimit * 1000)
			}
		}

		if options.memoryMiB == nil && options.memoryLimitMiB != nil {
			return nil, fmt.Errorf("must also specify non-zero MemoryMiB request when MemoryLimitMiB is specified")
		}
		if options.memoryMiB != nil {
			if *options.memoryMiB <= 0 {
				return nil, fmt.Errorf("the MemoryMiB request (%d) must be a positive number", *options.memoryMiB)
			}
			resBuilder.MemoryMb = uint32(*options.memoryMiB)
			if options.memoryLimitMiB != nil {
				if *options.memoryLimitMiB < *options.memoryMiB {
					return nil, fmt.Errorf("the MemoryMiB request (%d) cannot be higher than MemoryLimitMiB (%d)", *options.memoryMiB, *options.memoryLimitMiB)
				}
				resBuilder.MemoryMbMax = uint32(*options.memoryLimitMiB)
			}
		}

		if options.gpu != nil {
			gpuConfig, err := parseGPUConfig(*options.gpu)
			if err != nil {
				return nil, err
			}
			resBuilder.GpuConfig = gpuConfig
		}
		builder.Resources = resBuilder.Build()
	}

	secretIds := []string{}
	if options.secrets != nil {
		for _, secret := range *options.secrets {
			if secret != nil {
				secretIds = append(secretIds, secret.SecretID)
			}
		}
	}

	builder.SecretIds = secretIds
	if len(secretIds) > 0 {
		builder.ReplaceSecretIds = true
	}

	if options.volumes != nil {
		volumeMounts := []*pb.VolumeMount{}
		for mountPath, volume := range *options.volumes {
			if volume != nil {
				volumeMounts = append(volumeMounts, volumeToMountProto(mountPath, volume))
			}
		}
		builder.VolumeMounts = volumeMounts
		if len(volumeMounts) > 0 {
			builder.ReplaceVolumeMounts = true
		}
	}

	if options.retries != nil {
		builder.RetryPolicy = pb.FunctionRetryPolicy_builder{
			Retries:            uint32(options.retries.MaxRetries),
			BackoffCoefficient: options.retries.BackoffCoefficient,
			InitialDelayMs:     uint32(options.retries.InitialDelay / time.Millisecond),
			MaxDelayMs:         uint32(options.retries.MaxDelay / time.Millisecond),
		}.Build()
	}

	if options.maxContainers != nil {
		v := uint32(*options.maxContainers)
		builder.ConcurrencyLimit = &v
	}
	if options.bufferContainers != nil {
		v := uint32(*options.bufferContainers)
		builder.BufferContainers = &v
	}

	if options.scaledownWindow != nil {
		if *options.scaledownWindow < time.Second {
			return nil, fmt.Errorf("scaledownWindow must be at least 1 second, got %v", *options.scaledownWindow)
		}
		if (*options.scaledownWindow)%time.Second != 0 {
			return nil, fmt.Errorf("scaledownWindow must be a whole number of seconds, got %v", *options.scaledownWindow)
		}
		v := uint32((*options.scaledownWindow) / time.Second)
		builder.TaskIdleTimeoutSecs = &v
	}
	if options.timeout != nil {
		if *options.timeout < time.Second {
			return nil, fmt.Errorf("timeout must be at least 1 second, got %v", *options.timeout)
		}
		if (*options.timeout)%time.Second != 0 {
			return nil, fmt.Errorf("timeout must be a whole number of seconds, got %v", *options.timeout)
		}
		v := uint32((*options.timeout) / time.Second)
		builder.TimeoutSecs = &v
	}

	if options.maxConcurrentInputs != nil {
		v := uint32(*options.maxConcurrentInputs)
		builder.MaxConcurrentInputs = &v
	}
	if options.targetConcurrentInputs != nil {
		v := uint32(*options.targetConcurrentInputs)
		builder.TargetConcurrentInputs = &v
	}

	if options.batchMaxSize != nil {
		v := uint32(*options.batchMaxSize)
		builder.BatchMaxSize = &v
	}
	if options.batchWait != nil {
		v := uint64((*options.batchWait) / time.Millisecond)
		builder.BatchLingerMs = &v
	}

	if options.routingRegion != nil {
		builder.RoutingRegion = options.routingRegion
	}

	return builder.Build(), nil
}

// bindParameters processes the parameters and binds them to the function.
func bindParameters(
	ctx context.Context,
	client *Client,
	functionID string,
	options *functionOptions,
	schema []*pb.ClassParameterSpec,
	parameters map[string]any,
) (*pb.FunctionBindParamsResponse, error) {
	if options == nil {
		options = &functionOptions{}
	}

	mergedSecrets, err := mergeEnvIntoSecrets(ctx, client, options.env, options.secrets)
	if err != nil {
		return nil, err
	}

	mergedOptions := mergeFunctionOptions(options, &functionOptions{
		secrets: &mergedSecrets,
		env:     nil, // nil'ing env just to clarify it's not needed anymore
	})

	serializedParams, err := encodeParameterSet(schema, parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize parameters: %w", err)
	}

	functionOptions, err := buildFunctionOptionsProto(mergedOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to build function options: %w", err)
	}

	bindResp, err := client.cpClient.FunctionBindParams(ctx, pb.FunctionBindParamsRequest_builder{
		FunctionId:       functionID,
		SerializedParams: serializedParams,
		FunctionOptions:  functionOptions,
		EnvironmentName:  client.profile.Environment,
	}.Build())

	if err != nil {
		return nil, fmt.Errorf("failed to bind parameters: %w", err)
	}

	return bindResp, nil
}

func (f *Function) WithOptions(options *FunctionWithOptionsParams) *Function {
	if options == nil {
		options = &FunctionWithOptionsParams{}
	}

	var secretsPtr *[]*Secret
	if options.Secrets != nil {
		s := options.Secrets
		secretsPtr = &s
	}
	var volumesPtr *map[string]*Volume
	if options.Volumes != nil {
		v := options.Volumes
		volumesPtr = &v
	}
	var envPtr *map[string]string
	if options.Env != nil {
		e := options.Env
		envPtr = &e
	}

	merged := mergeFunctionOptions(f.options, &functionOptions{
		cpu:              options.CPU,
		cpuLimit:         options.CPULimit,
		memoryMiB:        options.MemoryMiB,
		memoryLimitMiB:   options.MemoryLimitMiB,
		gpu:              options.GPU,
		env:              envPtr,
		secrets:          secretsPtr,
		volumes:          volumesPtr,
		retries:          options.Retries,
		maxContainers:    options.MaxContainers,
		bufferContainers: options.BufferContainers,
		scaledownWindow:  options.ScaledownWindow,
		timeout:          options.Timeout,
		routingRegion:    options.RoutingRegion,
	})

	return &Function{
		FunctionID:     f.FunctionID,
		handleMetadata: f.handleMetadata,
		client:         f.client,
		options:        merged,
	}
}

func (f *Function) WithConcurrency(params *FunctionWithConcurrencyParams) *Function {
	if params == nil {
		params = &FunctionWithConcurrencyParams{}
	}

	merged := mergeFunctionOptions(f.options, &functionOptions{
		maxConcurrentInputs:    &params.MaxInputs,
		targetConcurrentInputs: params.TargetInputs,
	})

	return &Function{
		FunctionID:     f.FunctionID,
		handleMetadata: f.handleMetadata,
		client:         f.client,
		options:        merged,
	}
}

func (f *Function) WithBatching(params *FunctionWithBatchingParams) *Function {
	if params == nil {
		params = &FunctionWithBatchingParams{}
	}

	merged := mergeFunctionOptions(f.options, &functionOptions{
		batchMaxSize: &params.MaxBatchSize,
		batchWait:    &params.Wait,
	})

	return &Function{
		FunctionID:     f.FunctionID,
		handleMetadata: f.handleMetadata,
		client:         f.client,
		options:        merged,
	}
}

func (f *Function) Instance(ctx context.Context) (*Function, error) {
	if f == nil {
		return nil, fmt.Errorf("failed to dereference function")
	}

	boundFnID := f.FunctionID
	handleMetadata := f.handleMetadata

	if hasOptions(f.options) {
		bindResp, err := bindParameters(ctx, f.client, f.FunctionID, f.options, []*pb.ClassParameterSpec{}, map[string]any{})
		if err != nil {
			return nil, err
		}

		boundFnID = bindResp.GetBoundFunctionId()
		handleMetadata = bindResp.GetHandleMetadata()
	}

	return &Function{
		FunctionID:     boundFnID,
		handleMetadata: handleMetadata,
		client:         f.client,
		options:        &functionOptions{},
	}, nil
}

// FunctionGetCurrentStatsParams are options for Function.GetCurrentStats.
type FunctionGetCurrentStatsParams struct{}

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
	invocation, err := createControlPlaneInvocation(ctx, f.client.cpClient, f.client.logger, f.FunctionID, input, pb.FunctionCallInvocationType_FUNCTION_CALL_INVOCATION_TYPE_ASYNC)
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
func (f *Function) GetCurrentStats(ctx context.Context, params *FunctionGetCurrentStatsParams) (*FunctionStats, error) {
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

// GetWebURL returns the URL of a Function running as a Web Function.
// Returns empty string if this Function is not a Web Function.
func (f *Function) GetWebURL() string {
	return f.getWebURL()
}
