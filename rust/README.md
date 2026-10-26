# base122-fast

[![Crates.io](https://img.shields.io/crates/v/base122-fast.svg)](https://crates.io/crates/base122-fast)
[![Documentation](https://docs.rs/base122-fast/badge.svg)](https://docs.rs/base122-fast)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Base122](https://github.com/kevinAlbs/Base122) is a binary-to-text encoding
scheme that is significantly more space-efficient than Base64: it adds only
**~14% overhead** (Base64 adds 33%) while remaining valid UTF-8.

**base122-fast** is a high-performance implementation with optional
architecture-specific SIMD kernels, a fully branch-free decode path, and
`no_std` support.

## Features

- **Multi-Gbps throughput** on large payloads, single-threaded.
- **AVX2 kernels** for x86-64-v3 CPUs, enabled automatically when the crate
  is built with `-C target-cpu=x86-64-v3` (runtime-detected with scalar
  fallback).
- **NEON kernels** for `aarch64` (auto-enabled — ASIMD is mandatory in
  arm64-v8a, so no feature flag is needed).
- **Fully branch-free decoder** with vectorized structural validation; invalid
  input falls back to the scalar codec for exact error reporting.
- **`no_std`** with `alloc`: suitable for embedded targets and WASM.
- **Caller-managed buffers** (`encode_into`/`decode_into`) eliminate per-call
  allocation for repeated use.
- **Rigorously tested**: exhaustive small-input round-trips, randomized
  cross-checks of every SIMD kernel against the reference codec, and a
  portable simulation that verifies the NEON algorithms on any host.

## Quick Start

```toml
[dependencies]
base122-fast = "0.1"
```

```rust
use base122_fast::{decode, encode};

let data = b"hello world";
let encoded = encode(data);
let decoded = decode(&encoded).expect("invalid Base122");
assert_eq!(data, decoded.as_slice());
```

## API

| Function | Description |
| :--- | :--- |
| `encode(data: &[u8]) -> String` | Encodes bytes to a UTF-8 Base122 string. |
| `decode(s: &str) -> Result<Vec<u8>, &'static str>` | Decodes a Base122 string; returns the first structural error. |
| `encode_into(data, &mut Vec<u8>) -> usize` | Encodes into a caller-owned buffer; returns the byte length. |
| `decode_into(s: &str, &mut Vec<u8>) -> Result<usize, &'static str>` | Decodes into a caller-owned buffer; returns the byte length. |

The `*_into` variants clear the buffer and grow it only when its capacity is
insufficient, so repeated calls with similarly sized payloads perform **no
allocation**:

```rust
let mut buf = Vec::new();
let n = encode_into(data, &mut buf);   // first call may allocate
let n = encode_into(other, &mut buf);  // subsequent calls reuse the buffer
```

## Performance

Throughput depends on payload size, data entropy, and CPU capabilities.
Indicative single-threaded numbers for 1 MiB payloads (random bytes):

| Codec | Encode | Decode |
| :--- | :--- | :--- |
| Scalar SWAR baseline | ~0.9 GB/s | ~1.0 GB/s |
| x86-64-v3 (`-C target-cpu=x86-64-v3`) | ~2.5 GB/s | ~6.5 GB/s |
| NEON (aarch64) | ~1.7 GB/s | ~3.0 GB/s |

Escape-free data (e.g. repeated printable ASCII) encodes significantly faster —
the AVX2 and NEON encoders reach roughly 2.5–3.5 GB/s on such input. Buffer
reuse (`encode_into`/`decode_into`) removes allocation overhead entirely.

Run `cargo bench` to reproduce measurements on your own hardware.

## Platform Support

| Target | Acceleration | Activation |
| :--- | :--- | :--- |
| x86-64 baseline | Scalar SWAR (LLVM auto-vectorizes where beneficial) | always |
| x86-64-v3 (AVX2 + BMI2) | Dedicated AVX2 kernels | automatic when built with `-C target-cpu=x86-64-v3` (or `-C target-feature=+avx2`), runtime CPUID detection |
| aarch64 (arm64-v8a) | Dedicated NEON kernels | automatic (no feature flag) |
| Other targets | Scalar SWAR | always |

```sh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

The AVX2 kernels are gated on `cfg(target_feature = "avx2")`, so they are
compiled in automatically when the build target has AVX2 (e.g. via
`-C target-cpu=x86-64-v3` or `-C target-feature=+avx2`), and are still
runtime-detected with a scalar fallback for safety. The default build stays
portable to baseline x86-64.

## Implementation Notes

Base122 maps binary data to a UTF-8-safe subset of 122 non-control bytes. Six
ASCII codes (`\x00`, `\x0A`, `\x0D`, `\x22`, `\x26`, `\x5C`) are illegal and
are encoded as two-byte escape pairs.

The hot paths process fixed-size blocks in 64/128/256-bit registers:

- **Encode** expands 7-bit groups with SIMD shifts and table lookups,
  detects illegal groups with a nibble bit-matrix, and splices escape pairs
  branchlessly — blends with empty masks are no-ops on escape-free blocks, so
  no data-dependent branches are needed.
- **Decode** transforms 8-byte blocks into their 7 output bytes with a
  bit-gather, validating lead/continuation structure in parallel; anomalies
  hand off to the scalar codec for exact errors.
- The scalar codec uses SWAR (SIMD-within-a-register) techniques and
  branchless escape splicing, so even escape-heavy data avoids mispredicted
  branches.

The NEON kernels are generic over an operation trait with a portable
simulation, so the full algorithm is verified against the reference codec on
any host; the intrinsic implementation is a one-to-one transcription.

## License

MIT
