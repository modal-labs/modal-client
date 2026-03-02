/**
 * CBOR serialization utilities for Modal.
 *
 * This module encapsulates cbor-x usage with a consistent configuration
 * that ensures compatibility with the Python CBOR implementation.
 */

import { Encoder, Decoder, Options } from "cbor-x";

// Extend the Options interface to include undocumented options
interface ExtendedOptions extends Options {
  useTag259ForMaps?: boolean;
}

/**
 * Custom CBOR encoder configured for Modal's specific requirements.
 *
 * Configuration:
 * - mapsAsObjects: true - Encode Maps as Objects for compatibility
 * - useRecords: false - Disable record structures
 * - tagUint8Array: false - Don't tag Uint8Arrays (avoid tag 64)
 */
const encoderOptions: ExtendedOptions = {
  mapsAsObjects: true,
  useRecords: false,
  tagUint8Array: false,
  useTag259ForMaps: false,
};

const decoderOptions: ExtendedOptions = {
  mapsAsObjects: true,
  useRecords: false,
  tagUint8Array: false,
  useTag259ForMaps: false,
};

const encoder = new Encoder(encoderOptions);

/**
 * Custom CBOR decoder configured for Modal's specific requirements.
 */
const decoder = new Decoder(decoderOptions);

/**
 * Encode a JavaScript value to CBOR bytes.
 *
 * @param value - The JavaScript value to encode
 * @returns CBOR-encoded bytes as a Buffer
 */
export function cborEncode(value: any): Buffer {
  return encoder.encode(value);
}

/**
 * Decode CBOR bytes to a JavaScript value.
 *
 * @param data - The CBOR-encoded bytes to decode
 * @returns The decoded JavaScript value
 */
export function cborDecode(data: Buffer | Uint8Array): any {
  return decoder.decode(data);
}
