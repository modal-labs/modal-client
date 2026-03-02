package modal

// Test to make sure serialization behaviors are consistent.

import (
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

// Reproduce serialization test from the Python SDK.
// https://github.com/modal-labs/modal-client/blob/4c62d67ee2816146a2a5d42581f6fe7349fa1bf6/test/serialization_test.py
func TestParameterSerialization(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	schema := []*pb.ClassParameterSpec{
		pb.ClassParameterSpec_builder{Name: "foo", Type: pb.ParameterType_PARAM_TYPE_STRING}.Build(),
		pb.ClassParameterSpec_builder{Name: "i", Type: pb.ParameterType_PARAM_TYPE_INT}.Build(),
	}
	values := map[string]any{"i": 5, "foo": "bar"}

	serializedParams, err := encodeParameterSet(schema, values)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	byteData := []byte("\n\x0c\n\x03foo\x10\x01\x1a\x03bar\n\x07\n\x01i\x10\x02 \x05")
	g.Expect(serializedParams).Should(gomega.Equal(byteData))

	// Reverse the order of map keys and make sure it's deterministic.
	schema = []*pb.ClassParameterSpec{schema[1], schema[0]}
	serializedParams, err = encodeParameterSet(schema, values)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(serializedParams).Should(gomega.Equal(byteData))

	// Test with a parameter that has a default value.
	schema = []*pb.ClassParameterSpec{
		pb.ClassParameterSpec_builder{
			Name:         "x",
			Type:         pb.ParameterType_PARAM_TYPE_BYTES,
			HasDefault:   true,
			BytesDefault: []byte("\x00"),
		}.Build(),
	}
	serializedParams, err = encodeParameterSet(schema, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	byteData = []byte("\n\x08\n\x01x\x10\x042\x01\x00")
	g.Expect(serializedParams).Should(gomega.Equal(byteData))
}
