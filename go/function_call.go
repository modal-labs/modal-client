package modal

import (
	"context"
	"fmt"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
)

// FunctionCallService provides FunctionCall related operations.
type FunctionCallService interface {
	FromID(ctx context.Context, functionCallID string) (*FunctionCall, error)
}

type functionCallServiceImpl struct{ client *Client }

// FunctionCall references a Modal Function Call. Function Calls are
// Function invocations with a given input. They can be consumed
// asynchronously (see Get()) or cancelled (see Cancel()).
type FunctionCall struct {
	FunctionCallID string

	client *Client
}

// FromID looks up a FunctionCall by ID.
func (s *functionCallServiceImpl) FromID(ctx context.Context, functionCallID string) (*FunctionCall, error) {
	functionCall := FunctionCall{
		FunctionCallID: functionCallID,
		client:         s.client,
	}
	return &functionCall, nil
}

// FunctionCallGetParams are options for getting outputs from Function Calls.
type FunctionCallGetParams struct {
	// Timeout specifies the maximum duration to wait for the output.
	// If nil, no timeout is applied. If set to 0, it will check if the function
	// call is already completed.
	Timeout *time.Duration
}

// Get waits for the output of a FunctionCall.
// If timeout > 0, the operation will be cancelled after the specified duration.
func (fc *FunctionCall) Get(ctx context.Context, params *FunctionCallGetParams) (any, error) {
	if params == nil {
		params = &FunctionCallGetParams{}
	}
	invocation := controlPlaneInvocationFromFunctionCallID(fc.client.cpClient, fc.client.logger, fc.FunctionCallID)
	return invocation.awaitOutput(ctx, params.Timeout)
}

// FunctionCallCancelParams are options for cancelling Function Calls.
type FunctionCallCancelParams struct {
	TerminateContainers bool
}

// Cancel cancels a FunctionCall.
func (fc *FunctionCall) Cancel(ctx context.Context, params *FunctionCallCancelParams) error {
	if params == nil {
		params = &FunctionCallCancelParams{}
	}
	_, err := fc.client.cpClient.FunctionCallCancel(ctx, pb.FunctionCallCancelRequest_builder{
		FunctionCallId:      fc.FunctionCallID,
		TerminateContainers: params.TerminateContainers,
	}.Build())
	if err != nil {
		return fmt.Errorf("FunctionCallCancel failed: %w", err)
	}

	return nil
}
