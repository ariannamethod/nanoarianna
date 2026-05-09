#!/usr/bin/env python3
"""resonance_to_gguf.py — Convert Resonance 200M raw fp32 RS02 .bin → GGUF.

Input format (RS02, per resonance_forward.h:255-301):
    uint32   magic = 0x52533032 ("RS02")
    int32[9] header [E, B, T, H, D, R, M, V, _reserved]
    uint32   n_merges
    int32[3] × n_merges    BPE merge triples (only first 2 ints used)
    float32  × np          weights, ordered per resonance_forward.h:94-114 assign():
        tok_emb [V, E]
        for layer i in 0..B-1:
            wr_a [H, E, R]
            wr_b [H, R, T]
            gate [H]
            norm1 [E]
            wq [E, E], wk [E, E], wv [E, E], wo [E, E]
            norm2 [E]
            mlp_gate [M, E], mlp_up [M, E], mlp_down [E, M]
        norm_f [E]
        out_head [V, E]

Output: GGUF v3, readable by libnotorch's gguf_open() / gguf_dequant().
Both Q8_0 and Q4_K quantizers ported from janus_to_gguf.py (the GGUF
writer scaffold is reusable; the per-block tensor naming and walk are
fully rewritten for Resonance's layout per audit B.1 — Janus had
Echo + smear + backout, Resonance does not).

Python permitted (data-prep / shim conversion path per refined ban
2026-05-06). Deps: numpy. No torch needed (RS02 is raw fp32, no
state_dict gymnastics).

Usage:
    python3 resonance_to_gguf.py input.bin output.gguf --quant q8_0
    python3 resonance_to_gguf.py input.bin output.gguf --quant q4_k

Exit codes:
    0  success
    1  input format error (bad magic, header, etc.)
    2  argv error
"""
import argparse
import struct
import sys
import numpy as np


# ── GGUF type IDs (mirrors notorch/gguf.c:303-336 dequant switch) ──────────
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12

# GGUF metadata value types
GGUF_VT_UINT32  = 4
GGUF_VT_INT32   = 5
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL    = 7
GGUF_VT_STRING  = 8
GGUF_VT_ARRAY   = 9
GGUF_VT_UINT64  = 10

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGN   = 32

RS02_MAGIC = 0x52533032  # "RS02"


# ── Q8_0 quantizer (block 32 fp32 → fp16 scale + 32 int8) ─────────────────
def quantize_q8_0(weights: np.ndarray) -> bytes:
    n = weights.size
    if n % 32 != 0:
        raise ValueError(f"Q8_0 requires multiple of 32 elements, got {n}")
    w = weights.reshape(-1, 32).astype(np.float32)
    amax = np.maximum(np.abs(w).max(axis=1), 1e-30)
    scale = (amax / 127.0).astype(np.float32)
    qs = np.round(w / scale[:, None]).clip(-128, 127).astype(np.int8)
    out = bytearray()
    scale_fp16 = scale.astype(np.float16).view(np.uint16)
    for i in range(scale.shape[0]):
        out += struct.pack("<H", scale_fp16[i])
        out += qs[i].tobytes()
    return bytes(out)


# ── Q4_K quantizer (super-block 256 fp32 with 6-bit scales/mins + 4-bit nibbles) ──
# Block layout (144 bytes per super-block of 256 floats):
#   2 bytes  fp16 d (scale of scales)
#   2 bytes  fp16 dmin (scale of mins)
#  12 bytes  scales/mins packed 6+6 bits × 8 sub-blocks
# 128 bytes  4-bit nibbles (256 quantized values, two per byte)
def quantize_q4_k(weights: np.ndarray) -> bytes:
    n = weights.size
    if n % 256 != 0:
        raise ValueError(f"Q4_K requires multiple of 256 elements, got {n}")

    w = weights.reshape(-1, 256).astype(np.float32)
    nblocks = w.shape[0]
    out = bytearray()

    for blk in range(nblocks):
        x = w[blk].reshape(8, 32)              # 8 sub-blocks of 32
        sub_min  = x.min(axis=1)               # per-sub min
        sub_max  = x.max(axis=1)               # per-sub max
        sub_d    = (sub_max - sub_min) / 15.0  # 4-bit range
        sub_d    = np.where(sub_d < 1e-30, 1e-30, sub_d)

        # super-scale d, super-min dmin
        d    = float(sub_d.max())  if sub_d.max()  > 0 else 1.0
        dmin = float(sub_min.max()) if sub_min.max() > 0 else 0.0
        if abs(d) < 1e-30:    d = 1e-30
        if abs(dmin) < 1e-30: dmin = 1e-30

        scales_q = np.round(sub_d   / d   * 63.0).clip(0, 63).astype(np.uint8)
        mins_q   = np.round(sub_min / dmin * 63.0).clip(0, 63).astype(np.uint8)

        # write d, dmin (fp16)
        out += struct.pack("<H", np.float16(d).view(np.uint16))
        out += struct.pack("<H", np.float16(dmin).view(np.uint16))

        # 6-bit scales/mins packed (12 bytes for 8 sub-blocks ×6+6 bits)
        # Layout per llama.cpp's get_scale_min_k4:
        #   bytes 0-3: scales[0..3] low 6 bits (each in 6 bits packed tight)
        #   bytes 4-7: mins[0..3] low 6 bits
        #   bytes 8-11: high 4 bits of scales[4..7] | high 4 bits of mins[4..7]
        sm = bytearray(12)
        for i in range(4):
            sm[i]   = scales_q[i] & 0x3F
            sm[i+4] = mins_q[i]   & 0x3F
        for i in range(4):
            sm[i+8] = ((scales_q[i+4] & 0x0F) << 0) | ((mins_q[i+4] & 0x0F) << 4)
            # high 2 bits of scales[i+4] go into high 2 bits of sm[i] (bits 6-7)
            sm[i] |= ((scales_q[i+4] & 0x30) << 2)
            sm[i+4] |= ((mins_q[i+4] & 0x30) << 2)
        out += bytes(sm)

        # quantize each sub-block using its scale/min
        nibbles = bytearray(128)
        for sb in range(8):
            s = sub_d[sb]
            m = sub_min[sb]
            if abs(s) < 1e-30:
                s = 1e-30
            qv = np.round((x[sb] - m) / s).clip(0, 15).astype(np.uint8)
            for j in range(16):
                nibbles[sb*16 + j] = (qv[j*2] & 0x0F) | ((qv[j*2+1] & 0x0F) << 4)
        out += bytes(nibbles)

    return bytes(out)


# ── GGUF writer ────────────────────────────────────────────────────────────
def write_gguf_string(buf: bytearray, s: str):
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b))
    buf += b


def write_gguf_kv_uint32(buf: bytearray, key: str, val: int):
    write_gguf_string(buf, key)
    buf += struct.pack("<I", GGUF_VT_UINT32)
    buf += struct.pack("<I", val)


def write_gguf_kv_string(buf: bytearray, key: str, val: str):
    write_gguf_string(buf, key)
    buf += struct.pack("<I", GGUF_VT_STRING)
    write_gguf_string(buf, val)


def write_gguf_kv_array_int32(buf: bytearray, key: str, vals: list):
    write_gguf_string(buf, key)
    buf += struct.pack("<I", GGUF_VT_ARRAY)
    buf += struct.pack("<I", GGUF_VT_INT32)
    buf += struct.pack("<Q", len(vals))
    for v in vals:
        buf += struct.pack("<i", v)


def quantize_for(name: str, w: np.ndarray, quant: str) -> tuple:
    """Pick quant per-tensor.
    - Small tensors (gate, all norms) stay fp32 — too small for block quant.
    - Embeddings always Q8_0 (wide distribution eats Q4_K).
    - Rest use requested format.
    """
    n = w.size
    # Small tensors (<256 elements OR not multiple of 32 for Q8_0 / 256 for Q4_K)
    # keep as fp32 — gate (H=12), all norms (E=768 * 1d), etc.
    if n < 256 or n % 32 != 0 or "norm" in name or name.endswith(".gate") or name.endswith("_gate"):
        return GGML_TYPE_F32, w.astype("<f4").tobytes()
    if name in ("token_embd.weight", "output.weight"):
        return GGML_TYPE_Q8_0, quantize_q8_0(w)
    if quant == "q8_0":
        return GGML_TYPE_Q8_0, quantize_q8_0(w)
    if quant == "q4_k":
        if n % 256 != 0:
            return GGML_TYPE_Q8_0, quantize_q8_0(w)
        return GGML_TYPE_Q4_K, quantize_q4_k(w)
    raise ValueError(f"unknown quant {quant}")


# ── RS02 reader ────────────────────────────────────────────────────────────
def read_rs02(path: str):
    with open(path, "rb") as f:
        magic_bytes = f.read(4)
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != RS02_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x}, expected 0x{RS02_MAGIC:08x}")
        hdr = struct.unpack("<9i", f.read(36))
        E, B, T, H, D, R, M, V, _reserved = hdr
        n_merges = struct.unpack("<I", f.read(4))[0]
        merges = []
        for _ in range(n_merges):
            triple = struct.unpack("<3i", f.read(12))
            merges.append((triple[0], triple[1]))
        # Remainder is float32 weight blob
        np_count = (
            2 * V * E + E +
            B * (E + 3 * E * E + H * E * R + H * R * T + H + E * E + E + 3 * M * E)
        )
        weights = np.frombuffer(f.read(np_count * 4), dtype=np.float32)
        if weights.size != np_count:
            raise ValueError(f"weight count mismatch: got {weights.size}, expected {np_count}")
        cfg = dict(E=E, B=B, T=T, H=H, D=D, R=R, M=M, V=V)
    return cfg, merges, weights


# ── Walk Resonance state_dict in assign() order ────────────────────────────
def walk_tensors(cfg: dict, weights: np.ndarray):
    """Yield (name, shape, fp32_array) for each tensor in resonance_forward.h
    assign() order. Names are GGUF-llama-style for max compatibility with
    llama.cpp + notorch's loader."""
    E, B, T, H, R, M, V = cfg["E"], cfg["B"], cfg["T"], cfg["H"], cfg["R"], cfg["M"], cfg["V"]
    p = 0

    def take(n):
        nonlocal p
        chunk = weights[p:p+n]
        if chunk.size != n:
            raise ValueError(f"out of bytes at offset {p}, want {n}")
        p += n
        return chunk

    yield "token_embd.weight", (V, E), take(V * E)
    for i in range(B):
        prefix = f"blk.{i}."
        yield prefix + "rrpram_a",       (H, E, R), take(H * E * R)
        yield prefix + "rrpram_b",       (H, R, T), take(H * R * T)
        yield prefix + "rrpram_gate",    (H,),      take(H)
        yield prefix + "attn_norm.weight",  (E,),      take(E)
        yield prefix + "attn_q.weight",     (E, E),    take(E * E)
        yield prefix + "attn_k.weight",     (E, E),    take(E * E)
        yield prefix + "attn_v.weight",     (E, E),    take(E * E)
        yield prefix + "attn_output.weight",(E, E),    take(E * E)
        yield prefix + "ffn_norm.weight",   (E,),      take(E)
        yield prefix + "ffn_gate.weight",   (M, E),    take(M * E)
        yield prefix + "ffn_up.weight",     (M, E),    take(M * E)
        yield prefix + "ffn_down.weight",   (E, M),    take(E * M)
    yield "output_norm.weight", (E,),     take(E)
    yield "output.weight",      (V, E),   take(V * E)

    if p != weights.size:
        raise ValueError(f"trailing weight bytes: {weights.size - p} unconsumed")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--quant", type=str.lower, choices=["q8_0", "q4_k"], default="q8_0")
    args = ap.parse_args()

    cfg, merges, weights = read_rs02(args.input)
    print(f"[rs02] cfg = {cfg}")
    print(f"[rs02] n_merges = {len(merges)}")
    print(f"[rs02] weights = {weights.size} floats ({weights.size * 4 / 1024 / 1024:.1f} MB raw)")

    # Pre-compute tensor list + quantized payloads
    tensors = []  # list of (name, shape, ggml_type, payload bytes, byte_offset_placeholder)
    for name, shape, w in walk_tensors(cfg, weights):
        ttype, payload = quantize_for(name, w, args.quant)
        tensors.append([name, shape, ttype, payload, 0])
    print(f"[rs02] {len(tensors)} tensors prepared as {args.quant}")

    # Build header KV section
    head = bytearray()
    head += GGUF_MAGIC
    head += struct.pack("<I", GGUF_VERSION)
    head += struct.pack("<Q", len(tensors))   # tensor_count

    # KV count placeholder; rewrite after we count
    kv_count_pos = len(head)
    head += struct.pack("<Q", 0)

    kv_count = 0
    write_gguf_kv_string(head, "general.architecture", "resonance"); kv_count += 1
    write_gguf_kv_string(head, "general.name",         "resonance-200m"); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.embedding_length", cfg["E"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.block_count",      cfg["B"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.context_length",   cfg["T"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.attention.head_count",    cfg["H"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.attention.head_count_kv", cfg["H"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.feed_forward_length",     cfg["M"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.rrpram_rank",             cfg["R"]); kv_count += 1
    write_gguf_kv_uint32(head, "resonance.head_dim",                cfg["D"]); kv_count += 1
    write_gguf_kv_uint32(head, "tokenizer.ggml.vocab_size",         cfg["V"]); kv_count += 1
    # BPE merges as flattened int32 array (n_merges × 2)
    flat = []
    for a, b in merges:
        flat.extend([a, b])
    write_gguf_kv_array_int32(head, "resonance.bpe.merges", flat); kv_count += 1

    struct.pack_into("<Q", head, kv_count_pos, kv_count)

    # Write tensor metadata block (names + dims + type + offset placeholder)
    meta_block = bytearray()
    tensor_data_offset_placeholder = []
    for tn in tensors:
        name, shape, ttype, payload, _ = tn
        write_gguf_string(meta_block, name)
        meta_block += struct.pack("<I", len(shape))         # n_dims
        # dims in reverse (GGUF convention)
        for d in reversed(shape):
            meta_block += struct.pack("<Q", int(d))
        meta_block += struct.pack("<I", ttype)              # type
        tensor_data_offset_placeholder.append(len(meta_block))
        meta_block += struct.pack("<Q", 0)                  # offset (filled below)

    # Compute padding to GGUF_ALIGN
    pre_data_size = len(head) + len(meta_block)
    pad = (GGUF_ALIGN - (pre_data_size % GGUF_ALIGN)) % GGUF_ALIGN
    data_block = bytearray(b"\x00" * pad)

    # Now lay out tensor payloads with per-tensor 32-byte alignment
    for i, tn in enumerate(tensors):
        name, shape, ttype, payload, _ = tn
        # Align inside data block
        cur = len(data_block)
        align_pad = (GGUF_ALIGN - (cur % GGUF_ALIGN)) % GGUF_ALIGN
        data_block += b"\x00" * align_pad
        offset = len(data_block)
        struct.pack_into("<Q", meta_block, tensor_data_offset_placeholder[i], offset)
        data_block += payload

    print(f"[gguf] header={len(head)}B meta={len(meta_block)}B data={len(data_block)}B "
          f"total={len(head) + len(meta_block) + len(data_block)}B")

    with open(args.output, "wb") as f:
        f.write(head)
        f.write(meta_block)
        f.write(data_block)
    print(f"[gguf] wrote {args.output}")


if __name__ == "__main__":
    main()
