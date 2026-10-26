//! # base122-fast
//!
//! A high-performance [Base122] codec. Base122 is a binary-to-text encoding
//! with only **~14% overhead** over raw bytes (Base64 adds 33%), producing
//! valid UTF-8 output.
//!
//! The crate is `no_std` (requires `alloc`), and its hot paths scale from
//! scalar SWAR up to architecture-specific SIMD kernels:
//!
//! - **x86-64**: AVX2+BMI2 kernels, compiled in automatically when the
//!   build target has AVX2 (e.g. `-C target-cpu=x86-64-v3`), with runtime
//!   detection and scalar fallback.
//! - **aarch64**: dedicated NEON kernels, enabled automatically (ASIMD is
//!   mandatory in arm64-v8a).
//!
//! Encoding and decoding are fully branch-free on the vector paths (invalid
//! decode input falls back to the scalar codec for exact errors), and
//! [`encode_into`]/[`decode_into`] support caller-owned buffers so repeated
//! calls allocate nothing.
//!
//! # Example
//!
//! ```
//! let data = b"hello world";
//! let encoded = base122_fast::encode(data);
//! let decoded = base122_fast::decode(&encoded).expect("invalid Base122");
//! assert_eq!(data, decoded.as_slice());
//! ```
//!
//! [Base122]: https://github.com/kevinAlbs/Base122

#![no_std]
// The `target_feature` cfg values used below (avx2, bmi2) are checked at
// build time; allow the lint instead of declaring them in a build script.
#![allow(unexpected_cfgs)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

// AVX2 kernels are compiled in when the build target has AVX2 (e.g.
// `-C target-cpu=x86-64-v3`); they are still runtime-detected with a scalar
// fallback.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

#[cfg(target_arch = "aarch64")]
mod neon;
// Test builds compile the NEON kernels with a portable simulation so they are
// verified against the reference codec on any host.
#[cfg(all(test, not(target_arch = "aarch64")))]
mod neon;

mod scalar;
use scalar::*;

/// Encodes binary data into a Base122 string.
///
/// The output is valid UTF-8 and adds ~14% size overhead (Base64 adds 33%).
///
/// # Examples
///
/// ```
/// let data = b"data";
/// let encoded = base122_fast::encode(data);
/// assert!(!encoded.is_empty());
/// ```
pub fn encode(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    let mut out = Vec::new();
    encode_into(data, &mut out);
    unsafe { String::from_utf8_unchecked(out) }
}

/// Encodes binary data into a caller-provided buffer and returns the number of
/// bytes written. The buffer is cleared and grown only when its capacity is
/// insufficient, so repeated calls with similarly sized inputs perform no
/// allocation — the caller controls the buffer's lifecycle.
pub fn encode_into(data: &[u8], out: &mut Vec<u8>) -> usize {
    out.clear();
    let need = encoded_capacity(data.len());
    // `reserve` takes the number of additional elements beyond the current
    // length, not a target capacity.
    let additional = need.saturating_sub(out.len());
    if additional > 0 {
        out.reserve(additional);
    }
    if data.is_empty() {
        return 0;
    }

    let out_ptr = out.as_mut_ptr();

    let len = data.len();
    let ptr = data.as_ptr();

    let mut out_pos = 0usize;
    let mut i = 0usize;
    let mut pending_illegal_bits = 0u8;
    let mut has_pending_illegal = false;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if avx2::has_avx2() {
        // The AVX2 loop processes 28-byte blocks until one needs the scalar
        // pending-escape mechanism; resolve that block with the scalar path
        // and re-enter the AVX2 loop.
        while i + 29 <= len {
            if has_pending_illegal {
                unsafe {
                    scalar_block_28(
                        ptr,
                        len,
                        &mut i,
                        out_ptr,
                        &mut out_pos,
                        &mut pending_illegal_bits,
                        &mut has_pending_illegal,
                    )
                };
                continue;
            }
            let before = i;
            unsafe {
                avx2::encode_28_avx2(
                    data,
                    out_ptr,
                    &mut out_pos,
                    &mut i,
                    &mut pending_illegal_bits,
                    &mut has_pending_illegal,
                );
            }
            if i == before {
                // Adjacent-illegal block: process with the scalar path.
                unsafe {
                    scalar_block_28(
                        ptr,
                        len,
                        &mut i,
                        out_ptr,
                        &mut out_pos,
                        &mut pending_illegal_bits,
                        &mut has_pending_illegal,
                    )
                };
            }
        }
    }

    // NEON loop: 28-byte blocks (two 14-byte blocks interleaved to hide the
    // tbl/vext latencies; NEON is the mandatory arm64-v8a baseline, so this
    // is selected at compile time, not runtime).
    #[cfg(target_arch = "aarch64")]
    {
        while i + 29 <= len {
            if has_pending_illegal {
                unsafe {
                    scalar_block_28(
                        ptr,
                        len,
                        &mut i,
                        out_ptr,
                        &mut out_pos,
                        &mut pending_illegal_bits,
                        &mut has_pending_illegal,
                    )
                };
                continue;
            }
            let ok = unsafe {
                neon::encode_block_28::<neon::intr::Intr>(
                    data,
                    out_ptr,
                    &mut out_pos,
                    &mut i,
                )
            };
            if !ok {
                // Adjacent-illegal or pending block: process with the scalar
                // path, then continue vectorizing the rest.
                unsafe {
                    scalar_block_28(
                        ptr,
                        len,
                        &mut i,
                        out_ptr,
                        &mut out_pos,
                        &mut pending_illegal_bits,
                        &mut has_pending_illegal,
                    )
                };
            }
        }
    }

    // Scalar 28-byte loop (used directly when no vector path is available; a
    // no-op after either vector loop above has consumed all its blocks).
    while i + 29 <= len {
        unsafe {
            scalar_block_28(
                ptr,
                len,
                &mut i,
                out_ptr,
                &mut out_pos,
                &mut pending_illegal_bits,
                &mut has_pending_illegal,
            )
        };
    }

    let len = unsafe {
        encode_tail(
            data,
            i,
            out_ptr,
            &mut out_pos,
            pending_illegal_bits,
            has_pending_illegal,
        )
    };
    debug_assert!(
        len <= out.capacity(),
        "encode wrote {} bytes, buffer capacity {}",
        len,
        out.capacity()
    );
    unsafe { out.set_len(len) };
    len
}

/// Decodes a Base122 string back into its original binary form.
///
/// # Errors
///
/// Returns `Err` with a description of the first structural problem: an
/// invalid lead byte, a malformed or truncated escape sequence, or an
/// invalid continuation byte.
///
/// # Examples
///
/// ```
/// let data = b"data";
/// let encoded = base122_fast::encode(data);
/// let decoded = base122_fast::decode(&encoded).unwrap();
/// assert_eq!(data, &decoded[..]);
/// ```
pub fn decode(encoded: &str) -> Result<Vec<u8>, &'static str> {
    if encoded.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    decode_into(encoded, &mut out)?;
    Ok(out)
}

/// Decodes into a caller-provided buffer and returns the number of bytes
/// written. The buffer is cleared and grown only when its capacity is
/// insufficient, so repeated calls with similarly sized inputs perform no
/// allocation — the caller controls the buffer's lifecycle. On error the
/// buffer is left empty.
pub fn decode_into(encoded: &str, out: &mut Vec<u8>) -> Result<usize, &'static str> {
    out.clear();
    let need = decoded_capacity(encoded.len());
    // `reserve` takes the number of additional elements beyond the current
    // length, not a target capacity.
    let additional = need.saturating_sub(out.len());
    if additional > 0 {
        out.reserve(additional);
    }
    if encoded.is_empty() {
        return Ok(0);
    }

    let bytes = encoded.as_bytes();
    // Only the vector paths below use `len`/`ptr`; silence the warning on
    // builds without a vector path.
    #[allow(unused_variables)]
    let len = bytes.len();
    let out_ptr = out.as_mut_ptr();
    #[allow(unused_variables)]
    let ptr = bytes.as_ptr();
    let mut out_pos = 0usize;
    let mut acc = 0u64;
    let mut acc_bits = 0u32;
    #[allow(unused_mut)]
    let mut i = 0usize;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if avx2::has_avx2() {
        // The AVX2 loop decodes 8-byte blocks (7 output bytes each) until an
        // anomaly requires the scalar codec.
        let (new_i, last_lead, bad) = unsafe { avx2::decode_blocks_avx2(ptr, len, out_ptr) };
        if !bad && new_i > 0 {
            out_pos = new_i / 8 * 7;
            i = new_i;
            if last_lead {
                unsafe {
                    handle_last_lead(
                        ptr, &mut i, len, out_ptr, &mut out_pos, &mut acc, &mut acc_bits,
                    )
                }?;
            }
        }
    }

    // NEON window loop: 16-byte windows (14 output bytes each), selected at
    // compile time on arm64-v8a.
    #[cfg(target_arch = "aarch64")]
    {
        let (new_i, last_lead, bad) =
            unsafe { neon::decode_blocks::<neon::intr::Intr>(ptr, len, out_ptr) };
        if !bad && new_i > 0 {
            out_pos = new_i / 16 * 14;
            i = new_i;
            if last_lead {
                unsafe {
                    handle_last_lead(
                        ptr, &mut i, len, out_ptr, &mut out_pos, &mut acc, &mut acc_bits,
                    )
                }?;
            }
        }
    }

    unsafe { decode_tail(bytes, i, out_ptr, &mut out_pos, &mut acc, &mut acc_bits) }?;

    debug_assert!(
        out_pos <= out.capacity(),
        "decode wrote {} bytes, buffer capacity {}",
        out_pos,
        out.capacity()
    );
    unsafe { out.set_len(out_pos) };
    Ok(out_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_empty() {
        assert_eq!(encode(b""), "");
        assert_eq!(decode("").unwrap(), b"");
    }

    #[test]
    fn test_reused_buffer() {
        // encode_into/decode_into must produce identical output to encode/decode
        // and keep working across calls on a reused buffer.
        let data = b"hello world";
        let enc = encode(data);
        let mut out = Vec::new();
        for _ in 0..3 {
            let n = encode_into(data, &mut out);
            assert_eq!(&out[..n], enc.as_bytes());
        }
        // Reuse across different sizes (shrink then grow).
        encode_into(b"hi", &mut out);
        assert_eq!(&out[..], encode(b"hi").as_bytes());
        encode_into(b"0123456789", &mut out);
        assert_eq!(String::from_utf8_lossy(&out), encode(b"0123456789"));

        let mut dec = Vec::new();
        for _ in 0..3 {
            let n = decode_into(&enc, &mut dec).expect("decode failed");
            assert_eq!(&dec[..n], data);
        }
        // decode error leaves the buffer empty. U+0800 encodes as 0xE0 0xA0
        // 0x80, and 0xE0 is not a valid Base122 lead byte.
        assert!(decode_into("\u{800}x", &mut dec).is_err());
        assert!(dec.is_empty());
    }

    #[test]
    fn test_hello_world() {
        let data = b"hello world";
        let enc = encode(data);
        let dec = decode(&enc).expect("decoding failed");
        assert_eq!(dec, data);
    }

    #[test]
    fn test_single_byte_values() {
        for b in 0..=255u8 {
            let data = vec![b];
            let enc = encode(&data);
            let dec = decode(&enc).unwrap_or_else(|_| panic!("decoding failed for byte {}", b));
            assert_eq!(dec, data, "failed for byte {}", b);
        }
    }

    #[test]
    fn test_various_lengths_roundtrip() {
        for len in [
            0, 1, 2, 3, 6, 7, 8, 9, 14, 15, 16, 17, 31, 32, 33, 100, 255, 256, 511, 512,
        ] {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let enc = encode(&data);
            let dec = decode(&enc).expect("decoding failed");
            assert_eq!(dec, data, "roundtrip failed for length {}", len);
        }
    }

    #[test]
    fn test_all_illegal_bytes_handling() {
        let data = b"\x00\x0A\x0D\x22\x26\x5C";
        let enc = encode(data);
        let dec = decode(&enc).expect("decoding failed");
        assert_eq!(dec, data.as_ref());
    }

    #[test]
    fn test_mixed_content() {
        let data: Vec<u8> = (0..=255).collect();
        let enc = encode(&data);
        let dec = decode(&enc).expect("decoding failed");
        assert_eq!(dec, data);
    }

    #[test]
    fn test_repeated_illegal_bytes() {
        let data = vec![0u8; 100];
        let enc = encode(&data);
        let dec = decode(&enc).expect("decoding failed");
        assert_eq!(dec, data);
    }

    #[test]
    fn test_decode_invalid_lead_byte() {
        let invalid = vec![0x80u8];
        let s = unsafe { String::from_utf8_unchecked(invalid) };
        assert!(decode(&s).is_err());
        let invalid2 = vec![0xFFu8];
        let s2 = unsafe { String::from_utf8_unchecked(invalid2) };
        assert!(decode(&s2).is_err());
    }

    #[test]
    fn test_decode_truncated_escape() {
        let mut data = vec![0xC0u8];
        let s = unsafe { String::from_utf8_unchecked(data.clone()) };
        assert!(decode(&s).is_err());
        data.push(0x40);
        let s2 = unsafe { String::from_utf8_unchecked(data) };
        assert!(decode(&s2).is_err());
    }

    #[test]
    fn test_decode_invalid_continuation_byte() {
        let data = vec![0xC2u8, 0xFF];
        let s = unsafe { String::from_utf8_unchecked(data) };
        assert!(decode(&s).is_err());
    }

    #[test]
    fn test_shortened_at_end() {
        let data = vec![0u8];
        let enc = encode(&data);
        let dec = decode(&enc).expect("decode failed");
        assert_eq!(dec, data);
    }

    #[test]
    fn test_very_long_input() {
        use rand::Rng;
        const DATA_SIZE: usize = 5_000_000;
        let mut data = vec![0u8; DATA_SIZE];
        let mut rng = rand::rng();
        rng.fill_bytes(&mut data);
        let enc = encode(&data);
        let dec = decode(&enc).expect("decode failed");
        assert_eq!(dec, data);
    }

    /// Deterministic PRNG (splitmix64) so the fuzz tests below are fully
    /// reproducible on every run and platform.
    fn next_u64(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_bytes(seed: &mut u64, len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        for b in buf.iter_mut() {
            *b = (next_u64(seed) >> 33) as u8;
        }
        buf
    }

    /// Every 2-byte input (65,536 cases) round-trips through the public API,
    /// and every encode output is valid UTF-8.
    #[test]
    fn test_exhaustive_2byte() {
        let mut enc = Vec::new();
        let mut dec = Vec::new();
        let mut buf = [0u8; 2];
        for a in 0..=255u8 {
            buf[0] = a;
            for b in 0..=255u8 {
                buf[1] = b;
                let n = encode_into(&buf, &mut enc);
                let s = core::str::from_utf8(&enc[..n]).expect("encode output must be UTF-8");
                let m = decode_into(s, &mut dec).expect("decode failed");
                assert_eq!(&dec[..m], &buf[..], "roundtrip mismatch for {buf:?}");
            }
        }
    }

    /// Every 3-byte input (2^24 = 16.8M cases) round-trips. This is the
    /// smallest exhaustive set covering every escape arrangement: single
    /// illegals, adjacent pairs (two 7-bit groups sharing one byte), and the
    /// trailing-shortened group. Run with `--release` (~3 s).
    #[test]
    fn test_exhaustive_3byte() {
        let mut enc = Vec::new();
        let mut dec = Vec::new();
        let mut buf = [0u8; 3];
        for a in 0..=255u8 {
            buf[0] = a;
            for b in 0..=255u8 {
                buf[1] = b;
                for c in 0..=255u8 {
                    buf[2] = c;
                    let n = encode_into(&buf, &mut enc);
                    let s = core::str::from_utf8(&enc[..n]).expect("encode output must be UTF-8");
                    let m = decode_into(s, &mut dec).expect("decode failed");
                    assert_eq!(&dec[..m], &buf[..], "roundtrip mismatch for {buf:?}");
                }
            }
        }
    }

    /// `decode` must never panic, regardless of input. Random byte strings of
    /// every block-boundary length are fed to the public API, including
    /// strings that are not valid UTF-8. Any `Ok`/`Err` result is acceptable;
    /// a panic fails the test.
    #[test]
    fn test_decode_never_panics() {
        let mut seed = 0xD1B5_4A32_D192_ED03;
        let lens = [
            0usize, 1, 2, 3, 7, 8, 14, 15, 16, 28, 29, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256,
            511, 512, 1024, 4096,
        ];
        for &len in &lens {
            for _ in 0..300 {
                let s = unsafe { String::from_utf8_unchecked(rand_bytes(&mut seed, len)) };
                let _ = decode(&s);
            }
        }
        // Corrupted valid streams: truncate and flip bytes of real encodings,
        // then decode the damage — must still never panic.
        for len in [1usize, 8, 28, 57, 100, 1000] {
            for _ in 0..200 {
                let mut s = encode(&rand_bytes(&mut seed, len));
                let pos = (next_u64(&mut seed) % s.len() as u64) as usize;
                unsafe { s.as_bytes_mut()[pos] ^= 0xFF };
                let _ = decode(&s);
                let cut = s.floor_char_boundary((next_u64(&mut seed) % s.len() as u64) as usize);
                s.truncate(cut);
                let _ = decode(&s);
            }
        }
    }

    /// Encode output must always be valid UTF-8 (a Base122 invariant), checked
    /// with a strict `from_utf8` on random data of every size class.
    #[test]
    fn test_encode_valid_utf8() {
        let mut seed = 0x1234_5678_9ABC_DEF0;
        let mut enc = Vec::new();
        for len in [0usize, 1, 2, 7, 8, 13, 14, 15, 16, 28, 29, 32, 56, 57, 100, 255, 1000, 65536]
        {
            for _ in 0..25 {
                let data = rand_bytes(&mut seed, len);
                let n = encode_into(&data, &mut enc);
                core::str::from_utf8(&enc[..n]).expect("encode output must be valid UTF-8");
            }
        }
    }

    /// Round-trip every length 0..=512 with several random payloads, one of
    /// which is enriched with illegal bytes (the escape-heavy path). This
    /// exercises every block boundary of the 14/28-byte encode blocks and the
    /// 8/16-byte decode blocks through the public API.
    #[test]
    fn test_length_sweep() {
        let mut seed = 0xFEDC_BA98_7654_3210;
        let mut enc = Vec::new();
        let mut dec = Vec::new();
        for len in 0..=512usize {
            for trial in 0..4u32 {
                let mut data = rand_bytes(&mut seed, len);
                if trial == 3 {
                    for b in data.iter_mut() {
                        if *b % 2 == 0 {
                            *b = [0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C][(*b as usize) % 6];
                        }
                    }
                }
                let n = encode_into(&data, &mut enc);
                let s = core::str::from_utf8(&enc[..n]).expect("encode output must be UTF-8");
                let m = decode_into(s, &mut dec).expect("decode failed");
                assert_eq!(&dec[..m], &data[..], "len={len} trial={trial}");
            }
        }
    }
}
