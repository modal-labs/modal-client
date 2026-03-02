// Test to make sure serialization behaviors are consistent.

import { expect, test } from "vitest";

import { ClassParameterSpec, ParameterType } from "../proto/modal_proto/api";
import { encodeParameterSet } from "./cls";

// Reproduce serialization test from the Python SDK.
// https://github.com/modal-labs/modal-client/blob/4c62d67ee2816146a2a5d42581f6fe7349fa1bf6/test/serialization_test.py
test("ParameterSerialization", () => {
  let schema: ClassParameterSpec[] = [
    ClassParameterSpec.fromPartial({
      name: "foo",
      type: ParameterType.PARAM_TYPE_STRING,
    }),
    ClassParameterSpec.fromPartial({
      name: "i",
      type: ParameterType.PARAM_TYPE_INT,
    }),
  ];
  const values = { i: 5, foo: "bar" };

  let serializedParams = encodeParameterSet(schema, values);
  let byteData = new Uint8Array([
    10, 12, 10, 3, 102, 111, 111, 16, 1, 26, 3, 98, 97, 114, 10, 7, 10, 1, 105,
    16, 2, 32, 5,
  ]);
  expect(serializedParams).toEqual(byteData);

  // Reverse the order of map keys and make sure it's deterministic.
  const reversedSchema = [schema[1], schema[0]];
  const reversedSerializedParams = encodeParameterSet(reversedSchema, values);
  expect(reversedSerializedParams).toEqual(byteData);

  // Test with a parameter that has a default value.
  schema = [
    ClassParameterSpec.create({
      name: "x",
      type: ParameterType.PARAM_TYPE_BYTES,
      hasDefault: true,
      bytesDefault: new Uint8Array([0]),
    }),
  ];
  serializedParams = encodeParameterSet(schema, {});
  byteData = new Uint8Array([10, 8, 10, 1, 120, 16, 4, 50, 1, 0]);
  expect(serializedParams).toEqual(byteData);
});
