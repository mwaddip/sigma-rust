// Binary parse/serialize WASM bindings for BlockHeader.
//
// Rationale: the existing `BlockHeader.from_json` binding cannot parse
// Autolykos v2 headers because the JSON shape carries `powSolutions.d`
// and `powSolutions.w` as `null`, and sigma-rust's `DeserializeBigIntFrom`
// enum has only `String | SerdeJsonNumber` variants (no `Null`). This
// surfaces at mainnet h=417,792 (the Autolykos v1→v2 activation block).
//
// ergo-node-rust validates the chain through h=417,792 fine because it
// uses `Header::scorex_parse_bytes` (binary canonical) rather than the
// JSON path. The WASM binding's `from_json`-only exposure was an
// incidental gap; the ergots mainnet-validate harness is the first
// consumer that needs binary parity with the chain validator.
//
// These tests lock in:
//   1. Round-trip exact byte equality for an Autolykos v1 header
//      (h=417,791 — last v1 mainnet block).
//   2. Round-trip exact byte equality for an Autolykos v2 header
//      (h=417,792 — first v2 mainnet block; null d/w).
//   3. The previously-failing v2 case: `from_json` throws (as before),
//      `sigma_parse_bytes` succeeds.
//
// Fixtures captured 2026-05-25 from the user's local ergo-node REST
// surface (`/blocks/{id}/validation-fragments` → `.headerBytes`).

import { expect, assert } from "chai";

import * as ergo from "..";

let ergo_wasm;
beforeEach(async () => {
  ergo_wasm = await ergo;
});

// h=417,791 — last Autolykos v1 mainnet header. Version byte 0x01.
// Captured 2026-05-25 from /blocks/{id}/validation-fragments.
const H_417791_BYTES_HEX =
  "01cee668bdcb8d24cc25569e82d7500b2c56eefddfa4629834de1cf6c96b2bfc8047bfb6efa2ed041e51013e905f045ebf5af19e1a8510a98a516d355d31116908a191856a3fd7703a92b8f8c397b8a0100195e98f408654e419ed01d03fc5f959d2ab2775e281381451e114c2c6fd84c6aa2acc9f349575387dbcfb79a8e57ffc13d5c6f898f62e30075a8396918d62ba187d7e32cf3623e393f0ac1881e221bc5fb9515de633d0070c039bffbf19040300030e9662f3ed3448512424a273b3820ef83d3fee593cda88a115f344dc76ec4322038036568285bbb106e3c3fe5a9a333b3663a02c56021f1eeb7222a8580836557800000e3b018c0ec91a01a500c60349fb637b8c2580ee5c1413f375f55fa7788a93380f";

// h=417,792 — first Autolykos v2 mainnet header. Version byte 0x02.
// powSolutions.d and powSolutions.w are NULL in the JSON representation
// — the exact shape the JSON binding rejects. The binary representation
// has no nullable fields and parses fine.
const H_417792_BYTES_HEX =
  "02b21a1c00412b84033185f3cf6cdd345c4276628f3dda1e63b8502a4923c8e2bc8daa8b0dfcf7b1178ec4b1bd813258b41c3e51b4cc91b1466fa1c68116739ceed56f5d1ab2bdc87e238daeccf61f0085c20c14813891207346a5d024d9c5e415f7593065966c5f9bd60b64789e5b3abfe3e78568e9c220d72497fdf78200038113b698ac9bf62e4adbc9e8c0e8c5b3cc6fed0b91fd43f53622a98d2359497ec132ae3279d0b259066f98d580c0190000000002b3a06d6eaa8671431ba1db4dd427a77f75a5c2acbd71bfb725d38adc2b55f66916db4fbab447e127";

function hexToBytes(hex) {
  if (typeof hex !== "string" || hex.length % 2 !== 0) {
    throw new Error("hexToBytes: invalid hex input");
  }
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return out;
}

function bytesToHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

describe("BlockHeader.sigma_parse_bytes / sigma_serialize_bytes", () => {
  it("round-trips an Autolykos v1 header (h=417,791) byte-exactly", async () => {
    const bytes = hexToBytes(H_417791_BYTES_HEX);
    const hdr = ergo_wasm.BlockHeader.sigma_parse_bytes(bytes);
    assert(hdr != null);
    const round = hdr.sigma_serialize_bytes();
    expect(bytesToHex(round)).to.equal(H_417791_BYTES_HEX);
  });

  it("parses an Autolykos v2 header (h=417,792) where from_json would fail", async () => {
    const bytes = hexToBytes(H_417792_BYTES_HEX);
    const hdr = ergo_wasm.BlockHeader.sigma_parse_bytes(bytes);
    assert(hdr != null);
    const round = hdr.sigma_serialize_bytes();
    expect(bytesToHex(round)).to.equal(H_417792_BYTES_HEX);
  });

  it("preserves the version byte (0x01 for v1, 0x02 for v2)", async () => {
    const v1Bytes = hexToBytes(H_417791_BYTES_HEX);
    const v1Hdr = ergo_wasm.BlockHeader.sigma_parse_bytes(v1Bytes);
    const v1Round = v1Hdr.sigma_serialize_bytes();
    expect(v1Round[0]).to.equal(0x01);

    const v2Bytes = hexToBytes(H_417792_BYTES_HEX);
    const v2Hdr = ergo_wasm.BlockHeader.sigma_parse_bytes(v2Bytes);
    const v2Round = v2Hdr.sigma_serialize_bytes();
    expect(v2Round[0]).to.equal(0x02);
  });
});
