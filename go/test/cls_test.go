package test

import (
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func TestClsCall(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	cls, err := tc.Cls.FromName(ctx, "libmodal-test-support", "EchoCls", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	instance, err := cls.Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = instance.Method("nonexistent")
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.NotFoundError{}))

	function, err := instance.Method("echo_string")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, err := function.Remote(ctx, nil, map[string]any{"s": "hello"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))

	cls, err = tc.Cls.FromName(ctx, "libmodal-test-support", "EchoClsParametrized", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	instance, err = cls.Instance(ctx, map[string]any{"name": "hello-init"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	function, err = instance.Method("echo_parameter")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, _ = function.Remote(ctx, nil, nil)
	g.Expect(result).Should(gomega.Equal("output: hello-init"))
}

func TestClsNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	tc := newTestClient(t)

	_, err := tc.Cls.FromName(t.Context(), "libmodal-test-support", "NotRealClassName", nil)
	g.Expect(err).Should(gomega.BeAssignableToTypeOf(modal.NotFoundError{}))
}

func TestClsCallInputPlane(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	cls, err := tc.Cls.FromName(ctx, "libmodal-test-support", "EchoClsInputPlane", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	instance, err := cls.Instance(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	function, err := instance.Method("echo_string")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	result, err := function.Remote(ctx, nil, map[string]any{"s": "hello"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(result).Should(gomega.Equal("output: hello"))
}
