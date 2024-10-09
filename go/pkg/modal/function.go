package modal

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/protobuf/proto"
)

type RemoteFunction struct {
	id     string
	client ModalClientClient
}

func NewRemoteFunction(id string, client ModalClientClient) *RemoteFunction {
	return &RemoteFunction{id: id, client: client}
}

func (f *RemoteFunction) Call(ctx context.Context, args CombinedArgs) (*Value, error) {
	payloadValue, err := args.ToProto()
	if err != nil {
		return nil, fmt.Errorf("failed to convert args to proto: %v", err)
	}

	argsBytes, err := proto.Marshal(payloadValue)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal args: %v", err)
	}

	input := &FunctionPutInputsItem{
		Idx: 0,
		Input: &FunctionInput{
			FinalInput: true,
			DataFormat: DataFormat_DATA_FORMAT_PAYLOAD_VALUE,
			ArgsOneof: &FunctionInput_Args{
				Args: argsBytes,
			},
		},
	}

	funcMap, err := f.client.FunctionMap(ctx, &FunctionMapRequest{
		FunctionId:                 f.id,
		FunctionCallType:           FunctionCallType_FUNCTION_CALL_TYPE_UNARY,
		PipelinedInputs:            []*FunctionPutInputsItem{input},
		FunctionCallInvocationType: FunctionCallInvocationType_FUNCTION_CALL_INVOCATION_TYPE_ASYNC,
	})
	if err != nil {
		return nil, fmt.Errorf("function map failed: %v", err)
	}

	var funcOutputs *FunctionGetOutputsResponse
	backoff := time.Millisecond * 10
	maxBackoff := time.Second * 5
	for {
		funcOutputs, err = f.client.FunctionGetOutputs(ctx, &FunctionGetOutputsRequest{
			FunctionCallId: funcMap.FunctionCallId,
			LastEntryId:    "0-0",
			RequestedAt:    float64(time.Now().UnixNano()) / 1e9,
			Timeout:        1,
		})
		if err != nil {
			return nil, fmt.Errorf("function get outputs failed: %v", err)
		}

		if funcOutputs.NumUnfinishedInputs == 0 {
			break
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}

	if len(funcOutputs.Outputs) == 0 {
		return nil, fmt.Errorf("no outputs from function")
	}

	output := funcOutputs.Outputs[0]
	result := output.Result
	if result == nil {
		return nil, fmt.Errorf("no result in function output")
	}

	if result.Status == GenericResult_GENERIC_STATUS_SUCCESS {
		if output.DataFormat != DataFormat_DATA_FORMAT_PAYLOAD_VALUE {
			return nil, fmt.Errorf("unexpected data format: %v", output.DataFormat)
		}

		switch data := result.DataOneof.(type) {
		case *GenericResult_Data:
			var protoValue PayloadValue
			if err := proto.Unmarshal(data.Data, &protoValue); err != nil {
				return nil, fmt.Errorf("failed to unmarshal response proto: %v", err)
			}
			value := FromProto(&protoValue)
			if value == nil {
				return nil, fmt.Errorf("could not decode response proto")
			}
			return value, nil
		default:
			return nil, fmt.Errorf("could not decode response proto: unexpected type %T", data)
		}
	} else {
		status := GenericResult_GenericStatus(result.Status)
		return nil, fmt.Errorf("function call failed, status: %v\nexception: %s", status, result.Exception)
	}
}
