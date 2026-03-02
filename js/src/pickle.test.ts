import { describe, expect, test } from "vitest";
import { dumps, loads, type Protocol } from "./pickle";
import { Buffer } from "node:buffer";

test("PickleUnpickle", () => {
  const sample = {
    a: 1,
    b: [2, 3, true, null],
    c: new Uint8Array([4, 5, 6]),
    d: "hello 🎉",
  };
  for (const proto of [3, 4, 5] as Protocol[]) {
    const pkl = dumps(sample, proto);
    const back = loads(pkl);
    expect(back).toEqual(sample);
  }
});

// Python pickle compatibility tests (v4)
// Using `python -c "import pickle, base64; print(base64.b64encode(pickle.dumps(..., protocol=4)).decode())"`
const testCases = [
  {
    name: "MinusOne",
    b64: "gASVBgAAAAAAAABK/////y4=",
    expected: -1,
  },
  {
    // [b'1', b'2', b'3']
    name: "BytesList - uses MARK and APPENDS",
    b64: "gASVEQAAAAAAAABdlChDATGUQwEylEMBM5RlLg==",
    expected: [
      new Uint8Array([49]),
      new Uint8Array([50]),
      new Uint8Array([51]),
    ],
  },
  {
    name: "SimpleList",
    b64: "gASVCwAAAAAAAABdlChLAUsCSwNlLg==",
    expected: [1, 2, 3],
  },
  {
    name: "SimpleDict",
    b64: "gASVEQAAAAAAAAB9lCiMAWGUSwGMAWKUSwJ1Lg==",
    expected: { a: 1, b: 2 },
  },
  // Integer edge cases
  { name: "BININT1_0", b64: "gARLAC4=", expected: 0 },
  { name: "BININT1_255", b64: "gARL/y4=", expected: 255 },
  { name: "BININT2_-32768", b64: "gASVBgAAAAAAAABKAID//y4=", expected: -32768 },
  { name: "BININT2_-32767", b64: "gASVBgAAAAAAAABKAYD//y4=", expected: -32767 },
  { name: "BININT2_32767", b64: "gASVBAAAAAAAAABN/38u", expected: 32767 },
  {
    name: "BININT2_32768 (unsigned boundary)",
    b64: "gASVBAAAAAAAAABNAIAu",
    expected: 32768,
  },
  {
    name: "BININT2_36636",
    b64: "gASVBAAAAAAAAABNHI8u",
    expected: 36636,
  },
  {
    name: "BININT4_-36636",
    b64: "gASVBgAAAAAAAABK5HD//y4=",
    expected: -36636,
  },
  {
    name: "BININT2_65535 (max unsigned 16-bit)",
    b64: "gASVBAAAAAAAAABN//8u",
    expected: 65535,
  },
  {
    name: "BININT4_-65535",
    b64: "gASVBgAAAAAAAABKAQD//y4=",
    expected: -65535,
  },
  {
    name: "BININT4_-2147483648",
    b64: "gASVBgAAAAAAAABKAAAAgC4=",
    expected: -2147483648,
  },
  {
    name: "BININT4_2147483647",
    b64: "gASVBgAAAAAAAABK////fy4=",
    expected: 2147483647,
  },
];

describe("Python compatibility", () => {
  for (const { name, b64, expected } of testCases) {
    test(name, () => {
      const buf = Buffer.from(b64, "base64");
      const val = loads(new Uint8Array(buf));
      expect(val).toEqual(expected);
      expect(loads(dumps(val, 4))).toEqual(val);
    });
  }
});
