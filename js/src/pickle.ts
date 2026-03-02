// Minimal pickle codec in TypeScript supporting protocol 3, 4 and 5
// ============================================================
// Focus: JSON‑compatible primitives (null, bool, number, string, arrays, plain
//         objects) plus Uint8Array.  The encoder can *emit* protocol 3, 4 or 5
// (default 4).  The decoder can *read* any pickle whose first PROTO opcode is
// 3, 4 or 5 **provided it only uses the opcodes below**.  This is *not* a full
// Python pickler, but is more than good enough for lightweight data exchange.
// -------------------------------------------------------------
// Implemented opcodes
//   Generic:  PROTO, STOP, NONE, NEWTRUE, NEWFALSE
//   Numbers:  BININT1, BININT2, BININT4 (aka BININT), BINFLOAT
//   Text:     SHORT_BINUNICODE, BINUNICODE, BINUNICODE8
//   Bytes:    SHORT_BINBYTES,  BINBYTES,  BINBYTES8
//   Containers: EMPTY_LIST, APPEND, EMPTY_DICT, SETITEM, MARK, SETITEMS, APPENDS
//   Memo:     MEMOIZE   (≥4), BINPUT/LONG_BINPUT + BINGET/LONG_BINGET (≤3)
//   Frames:   FRAME (proto‑5) – we just skip the announced length.
// -------------------------------------------------------------

class PickleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PickleError";
  }
}

// ─── Opcode values (single‑byte) ─────────────────────────────
const enum Op {
  PROTO = 0x80, // PROTO n
  STOP = 0x2e, // .
  NONE = 0x4e, // N
  NEWTRUE = 0x88, // \x88
  NEWFALSE = 0x89, // \x89

  BININT1 = 0x4b, // K  (uint8)
  BININT2 = 0x4d, // M  (uint16 LE)
  BININT4 = 0x4a, // J  (int32 LE)
  BINFLOAT = 0x47, // G  (float64 BE)

  SHORT_BINUNICODE = 0x8c, // \x8c len(1) data
  BINUNICODE = 0x58, // X len(4) data
  BINUNICODE8 = 0x8d, // \x8d len(8) data (≥4)

  SHORT_BINBYTES = 0x43, // C len(1) data (≥3)
  BINBYTES = 0x42, // B len(4) data (≥3)
  BINBYTES8 = 0x8e, // \x8e len(8) data (≥4)

  EMPTY_LIST = 0x5d, // ]
  APPEND = 0x61, // a
  EMPTY_DICT = 0x7d, // }
  SETITEM = 0x73, // s
  MARK = 0x28, // (  (mark stack position)

  // Memo / frame machinery
  BINPUT = 0x71, // q  idx(1)
  LONG_BINPUT = 0x72, // r  idx(4)
  BINGET = 0x68, // h  idx(1)
  LONG_BINGET = 0x6a, // j  idx(4)
  MEMOIZE = 0x94, // \x94 (≥4)
  FRAME = 0x95, // \x95 size(8) (proto‑5)
  APPENDS = 0x65, // e
  SETITEMS = 0x75, // u
}

// ─── Binary helpers ─────────────────────────────────────────
class Writer {
  private out: number[] = [];
  byte(b: number) {
    this.out.push(b & 0xff);
  }
  bytes(arr: Uint8Array | number[]) {
    for (const b of arr) this.byte(b as number);
  }
  uint32LE(x: number) {
    this.byte(x);
    this.byte(x >>> 8);
    this.byte(x >>> 16);
    this.byte(x >>> 24);
  }
  uint64LE(n: number | bigint) {
    let v = BigInt(n);
    for (let i = 0; i < 8; i++) {
      this.byte(Number(v & 0xffn));
      v >>= 8n;
    }
  }
  float64BE(v: number) {
    const dv = new DataView(new ArrayBuffer(8));
    dv.setFloat64(0, v, false);
    this.bytes(new Uint8Array(dv.buffer));
  }
  toUint8(): Uint8Array {
    return new Uint8Array(this.out);
  }
}

class Reader {
  constructor(
    private buf: Uint8Array,
    private pos = 0,
  ) {}
  eof() {
    return this.pos >= this.buf.length;
  }
  byte() {
    return this.buf[this.pos++];
  }
  take(n: number) {
    const s = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return s;
  }
  uint32LE() {
    const b0 = this.byte(),
      b1 = this.byte(),
      b2 = this.byte(),
      b3 = this.byte();
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }
  uint64LE() {
    const lo = this.uint32LE() >>> 0;
    const hi = this.uint32LE() >>> 0;
    return hi * 2 ** 32 + lo;
  }
  int32LE() {
    const v = new DataView(
      this.buf.buffer,
      this.buf.byteOffset + this.pos,
      4,
    ).getInt32(0, true);
    this.pos += 4;
    return v;
  }
  float64BE() {
    const v = new DataView(
      this.buf.buffer,
      this.buf.byteOffset + this.pos,
      8,
    ).getFloat64(0, false);
    this.pos += 8;
    return v;
  }
}

// ─── Encoder ────────────────────────────────────────────────
export type Protocol = 3 | 4 | 5;

function encodeValue(val: any, w: Writer, proto: Protocol) {
  // null / bool ------------------------------------------------
  if (val === null || val === undefined) {
    w.byte(Op.NONE);
    return;
  }
  if (typeof val === "boolean") {
    w.byte(val ? Op.NEWTRUE : Op.NEWFALSE);
    return;
  }

  // number -----------------------------------------------------
  if (typeof val === "number") {
    if (Number.isInteger(val)) {
      if (val >= 0 && val <= 0xff) {
        w.byte(Op.BININT1);
        w.byte(val);
      } else if (val >= 0 && val <= 0xffff) {
        w.byte(Op.BININT2);
        w.byte(val & 0xff);
        w.byte((val >> 8) & 0xff);
      } else {
        w.byte(Op.BININT4);
        w.uint32LE(val >>> 0);
      }
    } else {
      w.byte(Op.BINFLOAT);
      w.float64BE(val);
    }
    maybeMemoize(w, proto);
    return;
  }

  // string -----------------------------------------------------
  if (typeof val === "string") {
    const utf8 = new TextEncoder().encode(val);
    if (proto >= 4 && utf8.length < 256) {
      w.byte(Op.SHORT_BINUNICODE);
      w.byte(utf8.length);
    } else if (proto >= 4 && utf8.length > 0xffff_ffff) {
      w.byte(Op.BINUNICODE8);
      w.uint64LE(utf8.length);
    } else {
      w.byte(Op.BINUNICODE);
      w.uint32LE(utf8.length);
    }
    w.bytes(utf8);
    maybeMemoize(w, proto);
    return;
  }

  // bytes / Uint8Array ----------------------------------------
  if (val instanceof Uint8Array) {
    const len = val.length;
    if (proto >= 4 && len < 256) {
      w.byte(Op.SHORT_BINBYTES);
      w.byte(len);
    } else if (proto >= 4 && len > 0xffff_ffff) {
      w.byte(Op.BINBYTES8);
      w.uint64LE(len);
    } else {
      w.byte(Op.BINBYTES);
      w.uint32LE(len);
    }
    w.bytes(val);
    maybeMemoize(w, proto);
    return;
  }

  // Array ------------------------------------------------------
  if (Array.isArray(val)) {
    w.byte(Op.EMPTY_LIST);
    maybeMemoize(w, proto);
    for (const item of val) {
      encodeValue(item, w, proto);
      w.byte(Op.APPEND);
    }
    return;
  }

  // plain object ----------------------------------------------
  if (typeof val === "object") {
    w.byte(Op.EMPTY_DICT);
    maybeMemoize(w, proto);
    for (const [k, v] of Object.entries(val)) {
      encodeValue(k, w, proto);
      encodeValue(v, w, proto);
      w.byte(Op.SETITEM);
    }
    return;
  }

  throw new PickleError(
    `The JS Modal SDK does not support encoding/pickling data of type ${typeof val}`,
  );
}

function maybeMemoize(w: Writer, proto: Protocol) {
  if (proto >= 4) {
    w.byte(Op.MEMOIZE);
  } // super-simple strategy: memo every value >=4
}

export function dumps(obj: any, protocol: Protocol = 4): Uint8Array {
  if (![3, 4, 5].includes(protocol))
    throw new PickleError(
      `The JS Modal SDK does not support pickle protocol version ${protocol}`,
    );
  const w = new Writer();
  w.byte(Op.PROTO);
  w.byte(protocol);
  if (protocol === 5) {
    // Emit a minimal zero‑length FRAME so CPython recognises proto‑5 content.
    w.byte(Op.FRAME);
    w.uint64LE(0);
  }
  encodeValue(obj, w, protocol);
  w.byte(Op.STOP);
  return w.toUint8();
}

// ─── Decoder ────────────────────────────────────────────────
export function loads(buf: Uint8Array): any {
  const r = new Reader(buf);
  const op0 = r.byte();
  if (op0 !== Op.PROTO) throw new PickleError("pickle missing PROTO header");
  const proto: Protocol = r.byte() as Protocol;
  if (![3, 4, 5].includes(proto))
    throw new PickleError(
      `The JS Modal SDK does not support pickle protocol version ${proto}`,
    );

  const stack: any[] = [];
  const memo: any[] = [];
  const tdec = new TextDecoder();

  function push(v: any) {
    stack.push(v);
  }
  function pop() {
    return stack.pop();
  }

  // If proto‑5 and next opcode is FRAME, consume size then continue.
  if (proto === 5 && buf[r["pos"]] === Op.FRAME) {
    r.byte(); // FRAME
    const size = r.uint64LE(); // we ignore the size and just stream‑read.
    void size; // silence tsclint
  }

  // Unique marker for stack operations (cannot be confused with user data)
  const MARK = Symbol("pickle-mark");

  while (!r.eof()) {
    const op = r.byte();
    switch (op) {
      case Op.STOP:
        return stack.pop();
      case Op.NONE:
        push(null);
        break;
      case Op.NEWTRUE:
        push(true);
        break;
      case Op.NEWFALSE:
        push(false);
        break;

      case Op.BININT1:
        push(r.byte());
        break;
      case Op.BININT2: {
        const lo = r.byte(),
          hi = r.byte();
        const n = (hi << 8) | lo;
        push(n);
        break;
      }
      case Op.BININT4: {
        push(r.int32LE());
        break;
      }
      case Op.BINFLOAT:
        push(r.float64BE());
        break;

      case Op.SHORT_BINUNICODE: {
        const n = r.byte();
        push(tdec.decode(r.take(n)));
        break;
      }
      case Op.BINUNICODE: {
        const n = r.uint32LE();
        push(tdec.decode(r.take(n)));
        break;
      }
      case Op.BINUNICODE8: {
        const n = r.uint64LE();
        push(tdec.decode(r.take(n)));
        break;
      }

      case Op.SHORT_BINBYTES: {
        const n = r.byte();
        push(r.take(n));
        break;
      }
      case Op.BINBYTES: {
        const n = r.uint32LE();
        push(r.take(n));
        break;
      }
      case Op.BINBYTES8: {
        const n = r.uint64LE();
        push(r.take(n));
        break;
      }

      case Op.EMPTY_LIST:
        push([]);
        break;
      case Op.APPEND: {
        const v = pop();
        const lst = pop();
        lst.push(v);
        push(lst);
        break;
      }
      case Op.EMPTY_DICT:
        push({});
        break;
      case Op.SETITEM: {
        const v = pop(),
          k = pop(),
          d = pop();
        d[k] = v;
        push(d);
        break;
      }

      // Memo handling ----------------------------------------
      case Op.MEMOIZE:
        memo.push(stack[stack.length - 1]);
        break;
      case Op.BINPUT:
        memo[r.byte()] = stack[stack.length - 1];
        break;
      case Op.LONG_BINPUT:
        memo[r.uint32LE()] = stack[stack.length - 1];
        break;
      case Op.BINGET:
        push(memo[r.byte()]);
        break;
      case Op.LONG_BINGET:
        push(memo[r.uint32LE()]);
        break;

      case Op.FRAME: {
        const _size = r.uint64LE();
        /* ignore */ break;
      }

      case Op.MARK:
        push(MARK);
        break;

      case Op.APPENDS: {
        // Pops all items after the last MARK and appends them to the list below the MARK
        // Find the last MARK
        const markIndex = stack.lastIndexOf(MARK);
        if (markIndex === -1) {
          throw new PickleError("APPENDS without MARK");
        }
        const lst = stack[markIndex - 1];
        if (!Array.isArray(lst)) {
          throw new PickleError("APPENDS expects a list below MARK");
        }
        const items = stack.slice(markIndex + 1);
        lst.push(...items);
        stack.length = markIndex - 1; // Remove everything after the list
        push(lst);
        break;
      }

      case Op.SETITEMS: {
        // Sets multiple key-value pairs in a dict after the last MARK
        // Find the last MARK
        const markIndex = stack.lastIndexOf(MARK);
        if (markIndex === -1) {
          throw new PickleError("SETITEMS without MARK");
        }
        const d = stack[markIndex - 1];
        if (typeof d !== "object" || d === null || Array.isArray(d)) {
          throw new PickleError("SETITEMS expects a dict below MARK");
        }
        const items = stack.slice(markIndex + 1);
        // Set key-value pairs (items come in pairs: key, value, key, value, ...)
        for (let i = 0; i < items.length; i += 2) {
          if (i + 1 < items.length) {
            d[items[i]] = items[i + 1];
          }
        }
        stack.length = markIndex - 1; // Remove everything after the dict
        push(d);
        break;
      }

      default:
        throw new PickleError(
          `The JS Modal SDK does not support decoding/unpickling this kind of data. Error: unsupported opcode 0x${op.toString(16)}`,
        );
    }
  }
  throw new PickleError("pickle stream ended without STOP");
}
