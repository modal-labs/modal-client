package test

import (
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func TestFunctionSpawn(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	function, err := tc.Functions.FromName(
		ctx,
		"libmodal-test-support", "echo_string", nil,
	)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	functionCall, err := function.Spawn(ctx, nil, map[string]any{"s": "hello"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, err := functionCall.Get(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))

	functionCall, err = tc.FunctionCalls.FromID(ctx, functionCall.FunctionCallID)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, err = functionCall.Get(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))

	sleep, err := tc.Functions.FromName(
		ctx,
		"libmodal-test-support", "sleep", nil,
	)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	functionCall, err = sleep.Spawn(ctx, nil, map[string]any{"t": 5})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = functionCall.Cancel(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = functionCall.Get(ctx, nil)
	g.Expect(err).Should(gomega.HaveOccurred())

	functionCall, err = sleep.Spawn(ctx, nil, map[string]any{"t": 5})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	timeout := 1 * time.Second
	_, err = functionCall.Get(ctx, &modal.FunctionCallGetParams{Timeout: &timeout})
	g.Expect(err).Should(gomega.HaveOccurred())
}

func TestFunctionCallGet0(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sleep, _ := tc.Functions.FromName(
		ctx,
		"libmodal-test-support", "sleep", nil,
	)

	functionCall, err := sleep.Spawn(ctx, []any{0.5}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	// Polling for output with timeout 0 should raise an error, since the
	// function call has not finished yet.
	timeout := 0 * time.Second
	_, err = functionCall.Get(ctx, &modal.FunctionCallGetParams{Timeout: &timeout})
	g.Expect(err).Should(gomega.HaveOccurred())

	// Wait for the function call to finish.
	result, err := functionCall.Get(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.BeNil())

	// Now we can get the result.
	result, err = functionCall.Get(ctx, &modal.FunctionCallGetParams{Timeout: &timeout})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.BeNil())
}
