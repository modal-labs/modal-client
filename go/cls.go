package modal

import (
	"context"
	"fmt"
	"sort"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// ClsService provides Cls related operations.
type ClsService interface {
	FromName(ctx context.Context, appName string, name string, params *ClsFromNameParams) (*Cls, error)
}

type clsServiceImpl struct{ client *Client }

// ClsWithOptionsParams represents runtime options for a Modal Cls.
type ClsWithOptionsParams = FunctionWithOptionsParams

// ClsWithConcurrencyParams represents concurrency configuration for a Modal Cls.
type ClsWithConcurrencyParams = FunctionWithConcurrencyParams

// ClsWithBatchingParams represents batching configuration for a Modal Cls.
type ClsWithBatchingParams = FunctionWithBatchingParams

// Cls represents a Modal class definition that can be instantiated with parameters.
// It contains metadata about the class and its methods.
type Cls struct {
	serviceFunctionID       string
	serviceOptions          *functionOptions
	serviceFunctionMetadata *pb.FunctionHandleMetadata

	client *Client
}

// ClsFromNameParams are options for client.Cls.FromName.
type ClsFromNameParams struct {
	Environment string
	// Version looks up a version-pinned Cls deployed at this App version.
	Version int
}

// FromName references a Cls from a deployed App by its name.
func (s *clsServiceImpl) FromName(ctx context.Context, appName string, name string, params *ClsFromNameParams) (*Cls, error) {
	if params == nil {
		params = &ClsFromNameParams{}
	}

	cls := Cls{
		client: s.client,
	}

	// Find class service function metadata. Service functions are used to implement class methods,
	// which are invoked using a combination of service function ID and the method name.
	serviceFunctionName := fmt.Sprintf("%s.*", name)
	serviceFunction, err := s.client.cpClient.FunctionGet(ctx, pb.FunctionGetRequest_builder{
		AppName:         appName,
		ObjectTag:       serviceFunctionName,
		EnvironmentName: firstNonEmpty(params.Environment, s.client.profile.Environment),
		AppVersion:      int32(params.Version),
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("class '%s/%s' not found", appName, name)}
	}
	if err != nil {
		return nil, fmt.Errorf("failed to look up class service function: %w", err)
	}

	// Validate that we only support parameter serialization format PROTO.
	parameterInfo := serviceFunction.GetHandleMetadata().GetClassParameterInfo()
	schema := parameterInfo.GetSchema()
	if len(schema) > 0 && parameterInfo.GetFormat() != pb.ClassParameterInfo_PARAM_SERIALIZATION_FORMAT_PROTO {
		return nil, fmt.Errorf("unsupported parameter format: %v", parameterInfo.GetFormat())
	}

	cls.serviceFunctionID = serviceFunction.GetFunctionId()
	cls.serviceFunctionMetadata = serviceFunction.GetHandleMetadata()

	return &cls, nil
}

// getServiceFunctionMetadata returns the class's service function metadata or an error if not set.
func (c *Cls) getServiceFunctionMetadata() (*pb.FunctionHandleMetadata, error) {
	if c.serviceFunctionMetadata == nil {
		return nil, fmt.Errorf("unexpected error: class has not been hydrated")
	}
	return c.serviceFunctionMetadata, nil
}

func (c *Cls) getSchema() ([]*pb.ClassParameterSpec, error) {
	metadata, err := c.getServiceFunctionMetadata()
	if err != nil {
		return nil, err
	}
	return metadata.GetClassParameterInfo().GetSchema(), nil
}

// Instance creates a new instance of the class with the provided parameters.
func (c *Cls) Instance(ctx context.Context, parameters map[string]any) (*ClsInstance, error) {
	schema, err := c.getSchema()
	if err != nil {
		return nil, err
	}

	metadata, err := c.getServiceFunctionMetadata()
	if err != nil {
		return nil, err
	}

	var functionID string
	var methodHandleMetadata map[string]*pb.FunctionHandleMetadata
	if len(schema) == 0 && !hasOptions(c.serviceOptions) {
		functionID = c.serviceFunctionID
		methodHandleMetadata = metadata.GetMethodHandleMetadata()
	} else {
		opts := c.serviceOptions
		if opts == nil {
			opts = &functionOptions{}
		}
		bindResp, err := c.bindParameters(ctx, parameters, opts)
		if err != nil {
			return nil, err
		}
		functionID = bindResp.GetBoundFunctionId()
		// Use the bound variant's per-method metadata so dynamic options such as
		// RoutingRegion (surfaced as input_plane_url/input_plane_region) take
		// effect at invocation time.
		methodHandleMetadata = bindResp.GetHandleMetadata().GetMethodHandleMetadata()
	}

	methods := make(map[string]*Function, len(methodHandleMetadata))
	for name, methodMetadata := range methodHandleMetadata {
		methods[name] = &Function{
			FunctionID:     functionID,
			handleMetadata: methodMetadata,
			client:         c.client,
		}
	}
	return &ClsInstance{methods: methods}, nil
}

// WithOptions overrides the static Function configuration at runtime.
func (c *Cls) WithOptions(params *ClsWithOptionsParams) *Cls {
	if params == nil {
		params = &ClsWithOptionsParams{}
	}

	var secretsPtr *[]*Secret
	if params.Secrets != nil {
		s := params.Secrets
		secretsPtr = &s
	}
	var volumesPtr *map[string]*Volume
	if params.Volumes != nil {
		v := params.Volumes
		volumesPtr = &v
	}
	var envPtr *map[string]string
	if params.Env != nil {
		e := params.Env
		envPtr = &e
	}

	merged := mergeFunctionOptions(c.serviceOptions, &functionOptions{
		cpu:              params.CPU,
		cpuLimit:         params.CPULimit,
		memoryMiB:        params.MemoryMiB,
		memoryLimitMiB:   params.MemoryLimitMiB,
		gpu:              params.GPU,
		env:              envPtr,
		secrets:          secretsPtr,
		volumes:          volumesPtr,
		retries:          params.Retries,
		maxContainers:    params.MaxContainers,
		bufferContainers: params.BufferContainers,
		scaledownWindow:  params.ScaledownWindow,
		timeout:          params.Timeout,
		routingRegion:    params.RoutingRegion,
	})

	return &Cls{
		serviceFunctionID:       c.serviceFunctionID,
		serviceOptions:          merged,
		serviceFunctionMetadata: c.serviceFunctionMetadata,
		client:                  c.client,
	}
}

// WithConcurrency creates an instance of the Cls with input concurrency enabled or overridden with new values.
func (c *Cls) WithConcurrency(params *ClsWithConcurrencyParams) *Cls {
	if params == nil {
		params = &ClsWithConcurrencyParams{}
	}

	merged := mergeFunctionOptions(c.serviceOptions, &functionOptions{
		maxConcurrentInputs:    &params.MaxInputs,
		targetConcurrentInputs: params.TargetInputs,
	})

	return &Cls{
		serviceFunctionID:       c.serviceFunctionID,
		serviceOptions:          merged,
		serviceFunctionMetadata: c.serviceFunctionMetadata,
		client:                  c.client,
	}
}

// WithBatching creates an instance of the Cls with dynamic batching enabled or overridden with new values.
func (c *Cls) WithBatching(params *ClsWithBatchingParams) *Cls {
	if params == nil {
		params = &ClsWithBatchingParams{}
	}

	merged := mergeFunctionOptions(c.serviceOptions, &functionOptions{
		batchMaxSize: &params.MaxBatchSize,
		batchWait:    &params.Wait,
	})

	return &Cls{
		serviceFunctionID:       c.serviceFunctionID,
		serviceOptions:          merged,
		serviceFunctionMetadata: c.serviceFunctionMetadata,
		client:                  c.client,
	}
}

// bindParameters processes the parameters and binds them to the class function.
func (c *Cls) bindParameters(ctx context.Context, parameters map[string]any, opts *functionOptions) (*pb.FunctionBindParamsResponse, error) {
	schema, err := c.getSchema()
	if err != nil {
		return nil, err
	}

	return bindParameters(ctx, c.client, c.serviceFunctionID, opts, schema, parameters)
}

// encodeParameterSet encodes the parameter values into a binary format.
func encodeParameterSet(schema []*pb.ClassParameterSpec, parameters map[string]any) ([]byte, error) {
	var encoded []*pb.ClassParameterValue

	for _, parameterSpec := range schema {
		paramValue, err := encodeParameter(parameterSpec, parameters[parameterSpec.GetName()])
		if err != nil {
			return nil, err
		}
		encoded = append(encoded, paramValue)
	}

	// Sort keys, identical to Python `SerializeToString(deterministic=True)`.
	sort.Slice(encoded, func(i, j int) bool {
		return encoded[i].GetName() < encoded[j].GetName()
	})
	return proto.Marshal(pb.ClassParameterSet_builder{Parameters: encoded}.Build())
}

// encodeParameter converts a Go value to a ParameterValue proto message
func encodeParameter(parameterSpec *pb.ClassParameterSpec, value any) (*pb.ClassParameterValue, error) {
	name := parameterSpec.GetName()
	paramType := parameterSpec.GetType()
	paramValue := pb.ClassParameterValue_builder{
		Name: name,
		Type: paramType,
	}.Build()

	switch paramType {
	case pb.ParameterType_PARAM_TYPE_STRING:
		if value == nil && parameterSpec.GetHasDefault() {
			value = parameterSpec.GetStringDefault()
		}
		strValue, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf("parameter '%s' must be a string", name)
		}
		paramValue.SetStringValue(strValue)

	case pb.ParameterType_PARAM_TYPE_INT:
		if value == nil && parameterSpec.GetHasDefault() {
			value = parameterSpec.GetIntDefault()
		}
		var intValue int64
		switch v := value.(type) {
		case int:
			intValue = int64(v)
		case int64:
			intValue = v
		case int32:
			intValue = int64(v)
		default:
			return nil, fmt.Errorf("parameter '%s' must be an integer", name)
		}
		paramValue.SetIntValue(intValue)

	case pb.ParameterType_PARAM_TYPE_BOOL:
		if value == nil && parameterSpec.GetHasDefault() {
			value = parameterSpec.GetBoolDefault()
		}
		boolValue, ok := value.(bool)
		if !ok {
			return nil, fmt.Errorf("parameter '%s' must be a boolean", name)
		}
		paramValue.SetBoolValue(boolValue)

	case pb.ParameterType_PARAM_TYPE_BYTES:
		if value == nil && parameterSpec.GetHasDefault() {
			value = parameterSpec.GetBytesDefault()
		}
		bytesValue, ok := value.([]byte)
		if !ok {
			return nil, fmt.Errorf("parameter '%s' must be a byte slice", name)
		}
		paramValue.SetBytesValue(bytesValue)

	default:
		return nil, fmt.Errorf("unsupported parameter type: %v", paramType)
	}

	return paramValue, nil
}

// ClsInstance represents an instantiated Modal class with bound parameters.
// It provides access to the class methods with the bound parameters.
type ClsInstance struct {
	methods map[string]*Function
}

// Method returns the Function with the given name from a ClsInstance.
func (c *ClsInstance) Method(name string) (*Function, error) {
	method, ok := c.methods[name]
	if !ok {
		return nil, NotFoundError{fmt.Sprintf("method '%s' not found on class", name)}
	}
	return method, nil
}
