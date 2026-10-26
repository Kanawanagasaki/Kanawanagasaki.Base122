    extern crate std;
    use super::*;
    use alloc::vec;
    
    use alloc::vec::Vec;

    fn encode_ref(data: &[u8]) -> Vec<u8> {
        crate::encode(data).into_bytes()
    }

    fn decode_ref(s: &[u8]) -> Result<Vec<u8>, &'static str> {
        let s = unsafe { crate::String::from_utf8_unchecked(s.to_vec()) };
        crate::decode(&s)
    }

    fn enc_cap(n: usize) -> usize {
        n.saturating_mul(8) / 7 + 32
    }
    fn dec_cap(n: usize) -> usize {
        n.saturating_mul(7) / 8 + 32
    }

    /// Full encode loop with a generic kernel: mirrors lib.rs's encode_into
    /// vector-path structure (kernel + scalar fallback + scalar tail).
    fn encode_with<O: NeonOps>(data: &[u8], out: &mut Vec<u8>) {
        let ptr = data.as_ptr();
        let len = data.len();
        let out_ptr = out.as_mut_ptr();
        let mut out_pos = 0usize;
        let mut i = 0usize;
        let mut pending_illegal_bits = 0u8;
        let mut has_pending_illegal = false;

        while i + 15 <= len {
            if has_pending_illegal {
                unsafe {
                    crate::scalar_block_14(
                        ptr, len, &mut i, out_ptr, &mut out_pos, &mut pending_illegal_bits,
                        &mut has_pending_illegal,
                    )
                };
                continue;
            }
            let ok = unsafe {
                encode_block::<O>(
                    data,
                    out_ptr,
                    &mut out_pos,
                    &mut i,
                    &mut pending_illegal_bits,
                    &mut has_pending_illegal,
                )
            };
            if !ok {
                unsafe {
                    crate::scalar_block_14(
                        ptr, len, &mut i, out_ptr, &mut out_pos, &mut pending_illegal_bits,
                        &mut has_pending_illegal,
                    )
                };
            }
        }
        // Scalar tail (14/7-byte blocks + bit tail), same as encode_into.
        let rest = unsafe {
            crate::encode_tail(
                data,
                i,
                out_ptr,
                &mut out_pos,
                pending_illegal_bits,
                has_pending_illegal,
            )
        };
        unsafe { out.set_len(rest) };
    }

    /// Decode into a fresh Vec via a generic kernel.
    fn decode_with_vec<O: NeonOps>(s: &[u8]) -> Result<Vec<u8>, &'static str> {
        let mut out = Vec::with_capacity(dec_cap(s.len()));
        let n = decode_with::<O>(s, &mut out)?;
        unsafe { out.set_len(n) };
        Ok(out)
    }

    /// Full decode loop with a generic kernel.
    fn decode_with<O: NeonOps>(s: &[u8], out: &mut Vec<u8>) -> Result<usize, &'static str> {
        let ptr = s.as_ptr();
        let len = s.len();
        let out_ptr = out.as_mut_ptr();
        let mut out_pos = 0usize;
        let mut acc = 0u64;
        let mut acc_bits = 0u32;
        let mut i = 0usize;

        let (new_i, last_lead, bad) = unsafe { decode_blocks::<O>(ptr, len, out_ptr) };
        if !bad && new_i > 0 {
            out_pos = new_i / 16 * 14;
            i = new_i;
            if last_lead {
                if i < len {
                    let b1 = unsafe { *ptr.add(i - 1) };
                    let b2 = unsafe { *ptr.add(i) };
                    if (b2 & 0xC0) != 0x80 {
                        return Err("Invalid continuation byte");
                    }
                    unsafe {
                        crate::push7_scalar(
                            out_ptr, &mut out_pos, &mut acc, &mut acc_bits,
                            ((b1 & 1) << 6) | (b2 & 0x3F),
                        )
                    };
                    i += 1;
                } else {
                    return Err("Unexpected end of input");
                }
            }
        }
        unsafe { crate::decode_tail(s, i, out_ptr, &mut out_pos, &mut acc, &mut acc_bits) }?;
        unsafe { out.set_len(0) };
        Ok(out_pos)
    }

    #[test]
    fn test_sim_expand_matches_reference() {
        use rand::Rng;
        use sim::Sim;
        let mut rng = rand::rng();
        for _ in 0..2000 {
            let mut data = vec![0u8; 14];
            rng.fill_bytes(&mut data);
            let mut out = Vec::with_capacity(enc_cap(256));
            encode_with::<Sim>(&data, &mut out);
            assert_eq!(out, encode_ref(&data), "encode mismatch");
        }
    }

    #[test]
    fn test_sim_decode_matches_reference() {
        use rand::Rng;
        use sim::Sim;
        let mut rng = rand::rng();
        for _ in 0..2000 {
            let mut data = vec![0u8; 14];
            rng.fill_bytes(&mut data);
            let enc = encode_ref(&data);
            let mut out = Vec::with_capacity(enc_cap(256));
            let n = decode_with::<Sim>(&enc, &mut out).expect("decode failed");
            unsafe { out.set_len(n) };
            assert_eq!(out, data, "decode mismatch");
        }
    }

    #[test]
    fn test_sim_roundtrip_lengths() {
        use sim::Sim;
        for len in [0usize, 1, 2, 7, 8, 13, 14, 15, 16, 28, 29, 30, 31, 32, 33, 100, 1000] {
            let data: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) % 251) as u8).collect();
            let mut out = Vec::with_capacity(enc_cap(len));
            encode_with::<Sim>(&data, &mut out);
            assert_eq!(out, encode_ref(&data), "encode mismatch len {}", len);
            let mut dec = Vec::with_capacity(dec_cap(out.len()));
            let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
            unsafe { dec.set_len(n) };
            assert_eq!(dec, data, "roundtrip mismatch len {}", len);
        }
    }

    #[test]
    fn test_sim_encode_seed30_block() {
        use sim::Sim;
        // Regression: a block with many illegal groups in awkward positions.
        let data = [
            0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C, 0x7F, 0x00, 0x01, 0x02, 0x0A, 0x0D, 0x22, 0x26,
        ];
        let mut out = Vec::with_capacity(enc_cap(256));
        encode_with::<Sim>(&data, &mut out);
        assert_eq!(out, encode_ref(&data));
        let mut dec = Vec::with_capacity(dec_cap(512));
        let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
        unsafe { dec.set_len(n) };
        assert_eq!(dec, data.as_slice());
    }

    /// The 28-byte double-block kernel must produce the same output as the
    /// reference codec, including blocks with escapes at the boundaries.
    #[test]
    fn test_sim_encode28_matches_reference() {
        use sim::Sim;
        let mut x = 0x1234_5678_9abc_def0u64;
        let mut buf = vec![0u8; 57]; // 2 blocks of 28 + 1 guard byte (the kernel reads 29 bytes)
        let mut out = Vec::with_capacity(enc_cap(256));
        // Deterministic pseudo-random data, including illegal values.
        for iter in 0..4000 {
            for b in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793405).wrapping_add(1442695040888963407);
                *b = (x >> 33) as u8;
                if iter % 7 == 0 && *b % 13 == 0 {
                    *b = [0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C][(*b as usize) % 6];
                }
            }
            // Drive encode_block_28 directly through the same pipeline the
            // vector loop uses (28-byte blocks, scalar fallback, then the
            // shared tail).
            out.clear();
            let mut o = 0usize;
            let mut i = 0usize;
            let mut pending_illegal_bits = 0u8;
            let mut has_pending_illegal = false;
            while i + 29 <= buf.len() {
                if has_pending_illegal {
                    // Pending escape from the previous block: the scalar path
                    // consumes it, then we resume vectorizing.
                    unsafe {
                        crate::scalar_block_28(
                            buf.as_ptr(),
                            buf.len(),
                            &mut i,
                            out.as_mut_ptr(),
                            &mut o,
                            &mut pending_illegal_bits,
                            &mut has_pending_illegal,
                        )
                    };
                    continue;
                }
                let ok = unsafe { encode_block_28::<Sim>(&buf, out.as_mut_ptr(), &mut o, &mut i) };
                if !ok {
                    // Fall back: scalar 28-byte block.
                    unsafe {
                        crate::scalar_block_28(
                            buf.as_ptr(),
                            buf.len(),
                            &mut i,
                            out.as_mut_ptr(),
                            &mut o,
                            &mut pending_illegal_bits,
                            &mut has_pending_illegal,
                        )
                    };
                }
            }
            let o = unsafe {
                crate::encode_tail(
                    &buf,
                    i,
                    out.as_mut_ptr(),
                    &mut o,
                    pending_illegal_bits,
                    has_pending_illegal,
                )
            };
            unsafe { out.set_len(o) };
            assert_eq!(out, encode_ref(&buf), "iter {iter}");
        }
    }

    /// The vector kernel must actually run for blocks the scalar codec does
    /// not need (a bug where the group extraction never matched the reference
    /// made it return false for every input, silently falling back to scalar
    /// on every block).
    #[test]
    fn test_sim_encode28_uses_neon_path() {
        use sim::Sim;
        for &v in &[0x61u8, 0x7F, 0x41, 0x30] {
            let buf = vec![v; 57];
            let mut out = vec![0u8; 128];
            let mut o = 0usize;
            let mut i = 0usize;
            // 0x61/0x7F streams contain no illegal group; 0x41/0x30 contain
            // one, but not adjacent and not at a block boundary — the kernel
            // must accept all four outright.
            let ok = unsafe { encode_block_28::<Sim>(&buf, out.as_mut_ptr(), &mut o, &mut i) };
            assert!(ok, "0x{v:02X}: block must use the NEON path");
            assert_eq!(i, 28);
            assert_eq!(o, 32);
        }
        // All-zero bytes are illegal and adjacent everywhere: scalar fallback.
        let buf = vec![0u8; 57];
        let mut out = vec![0u8; 128];
        let mut o = 0usize;
        let mut i = 0usize;
        let ok = unsafe { encode_block_28::<Sim>(&buf, out.as_mut_ptr(), &mut o, &mut i) };
        assert!(!ok, "all-zero block must fall back to scalar");
    }

    /// Exhaustive: every 3-byte window (random enough to cover all escape
    /// positions) through encode + decode.
    #[test]
    fn test_sim_exhaustive_small() {
        use sim::Sim;
        let mut buf = vec![0u8; 3];
        let mut out = Vec::with_capacity(enc_cap(256));
        let mut dec = Vec::with_capacity(dec_cap(512));
        let mut iter = 0u32;
        for a in 0..=255u8 {
            buf[0] = a;
            for b in 0..=255u8 {
                buf[1] = b;
                for c in 0..=255u8 {
                    buf[2] = c;
                    encode_with::<Sim>(&buf, &mut out);
                    assert_eq!(out, encode_ref(&buf), "encode mismatch {:?}", buf);
                    let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
                    unsafe { dec.set_len(n) };
                    assert_eq!(dec, buf, "roundtrip mismatch {:?}", buf);
                    iter += 1;
                    if iter >= 60_000 {
                        return;
                    }
                }
            }
        }
    }

    /// Invalid streams: the sim decoder must agree with the reference decoder
    /// on the error outcome (both Err, or both the same bytes).
    #[test]
    fn test_sim_decode_invalid_agrees_with_reference() {
        use rand::Rng;
        use sim::Sim;
        let mut rng = rand::rng();
        let mut enc = Vec::with_capacity(enc_cap(256));
        for _ in 0..1000 {
            let mut data = vec![0u8; 14];
            rng.fill_bytes(&mut data);
            let good = encode_ref(&data);
            // Mutate 1-3 bytes at random positions.
            let mut stream = good.clone();
            let mut n = 1 + rand::random::<u32>() % 3;
            while n > 0 {
                let p = rand::random::<u32>() as usize % stream.len();
                stream[p] = rand::random::<u8>();
                n -= 1;
            }
            enc.clear();
            encode_with::<Sim>(&stream, &mut enc);
            let n = decode_with_vec::<Sim>(&enc).ok();
            let m = decode_ref(&enc).ok();
            match (n, m) {
                (None, None) => {}
                (Some(a), Some(b)) => assert_eq!(a, b, "decode mismatch on {:?}", enc),
                _ => panic!("decode outcome mismatch on {:?}", enc),
            }
        }
    }


    /// Large-scale intrinsic cross-check: encode/decode vs the reference
    /// codec across many sizes (aarch64-only).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_intr_large_crosscheck() {
        use rand::Rng;
        use intr::Intr;
        let mut rng = rand::rng();
        for round in 0..400 {
            let len = 256 + (rand::random::<u32>() as usize % 4096);
            let mut data = vec![0u8; len];
            rng.fill_bytes(&mut data);
            let mut out = Vec::with_capacity(enc_cap(len));
            encode_with::<Intr>(&data, &mut out);
            assert_eq!(out, encode_ref(&data), "encode mismatch round={} len={}", round, len);
            let mut dec = Vec::with_capacity(dec_cap(out.len()));
            let n = decode_with::<Intr>(&out, &mut dec).expect("decode failed");
            unsafe { dec.set_len(n) };
            assert_eq!(dec, data, "roundtrip mismatch round={} len={}", round, len);
        }
    }

    /// The intrinsic kernels must at least compile (aarch64-only).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_intr_roundtrip() {
        use rand::Rng;
        use intr::Intr;
        let mut rng = rand::rng();
        for _ in 0..200 {
            let mut data = vec![0u8; 100];
            rng.fill_bytes(&mut data);
            let mut out = Vec::with_capacity(enc_cap(100));
            encode_with::<Intr>(&data, &mut out);
            assert_eq!(out, encode_ref(&data));
            let mut dec = Vec::with_capacity(dec_cap(200));
            let n = decode_with::<Intr>(&out, &mut dec).expect("decode failed");
            unsafe { dec.set_len(n) };
            assert_eq!(dec, data);
        }
    }

    /// Exhaustive 2-byte roundtrip through the intrinsic NEON kernels
    /// (aarch64-only). Two bytes never reach the vector blocks, so this pins
    /// the escape/tail path on real hardware.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_intr_exhaustive_2byte() {
        use intr::Intr;
        let mut buf = vec![0u8; 2];
        let mut out = Vec::with_capacity(enc_cap(256));
        let mut dec = Vec::with_capacity(dec_cap(512));
        for a in 0..=255u8 {
            buf[0] = a;
            for b in 0..=255u8 {
                buf[1] = b;
                encode_with::<Intr>(&buf, &mut out);
                assert_eq!(out, encode_ref(&buf), "encode mismatch [{a:02x}, {b:02x}]");
                let n = decode_with::<Intr>(&out, &mut dec).expect("decode failed");
                unsafe { dec.set_len(n) };
                assert_eq!(dec, buf, "roundtrip mismatch [{a:02x}, {b:02x}]");
            }
        }
    }

    /// Length sweep through the intrinsic NEON kernels (aarch64-only):
    /// every length 1..=64 crosses the 14-byte encode block and 16-byte
    /// decode window boundaries.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_intr_length_sweep() {
        use rand::Rng;
        use intr::Intr;
        let mut rng = rand::rng();
        for len in 1..=64usize {
            for _ in 0..10 {
                let mut data = vec![0u8; len];
                rng.fill_bytes(&mut data);
                let mut out = Vec::with_capacity(enc_cap(len));
                encode_with::<Intr>(&data, &mut out);
                assert_eq!(out, encode_ref(&data), "encode mismatch len={len}");
                let mut dec = Vec::with_capacity(dec_cap(out.len()));
                let n = decode_with::<Intr>(&out, &mut dec).expect("decode failed");
                unsafe { dec.set_len(n) };
                assert_eq!(dec, data, "roundtrip mismatch len={len}");
            }
        }
    }

    /// Stress-test: every length from 0..=64 with random data, comparing
    /// the SIM NEON codec against the scalar reference.
    #[test]
    fn test_sim_stress_all_lengths() {
        use rand::Rng;
        use sim::Sim;
        let mut rng = rand::rng();
        for len in 0..=128usize {
            for _trial in 0..20 {
                let mut data = vec![0u8; len];
                rng.fill_bytes(&mut data);
                let mut out = Vec::with_capacity(enc_cap(len));
                encode_with::<Sim>(&data, &mut out);
                let ref_enc = encode_ref(&data);
                assert_eq!(out, ref_enc, "encode mismatch len={}", len);
                let mut dec = Vec::with_capacity(dec_cap(out.len()));
                let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
                unsafe { dec.set_len(n) };
                assert_eq!(dec, data, "roundtrip mismatch len={}", len);
            }
        }
    }

    /// Stress-test: complete 2-byte exhaustive roundtrip (all 65536 values).
    #[test]
    fn test_sim_exhaustive_2byte() {
        use sim::Sim;
        let mut buf = vec![0u8; 2];
        let mut out = Vec::with_capacity(enc_cap(256));
        let mut dec = Vec::with_capacity(dec_cap(512));
        for a in 0..=255u8 {
            buf[0] = a;
            for b in 0..=255u8 {
                buf[1] = b;
                encode_with::<Sim>(&buf, &mut out);
                let ref_enc = encode_ref(&buf);
                assert_eq!(out, ref_enc, "encode mismatch [{:02x}, {:02x}]", a, b);
                let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
                unsafe { dec.set_len(n) };
                assert_eq!(dec, buf, "roundtrip mismatch [{:02x}, {:02x}]", a, b);
            }
        }
    }

    /// Stress-test: all single-byte values repeated 16 times (covers many
    /// NEON block paths with homogeneous illegal/clean groups).
    #[test]
    fn test_sim_repeated_byte_blocks() {
        use sim::Sim;
        for val in 0..=255u8 {
            for rep in [1usize, 14, 15, 16, 28, 29, 30, 31, 32, 42, 56, 64, 100] {
                let data = vec![val; rep];
                let mut out = Vec::with_capacity(enc_cap(rep));
                encode_with::<Sim>(&data, &mut out);
                let ref_enc = encode_ref(&data);
                assert_eq!(out, ref_enc, "encode mismatch val=0x{:02x} rep={}", val, rep);
                let mut dec = Vec::with_capacity(dec_cap(out.len()));
                let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
                unsafe { dec.set_len(n) };
                assert_eq!(dec, data, "roundtrip mismatch val=0x{:02x} rep={}", val, rep);
            }
        }
    }

    /// Stress-test: alternating illegal + clean bytes to exercise the
    /// adjacent-illegal detection and scalar fallback.
    #[test]
    fn test_sim_alternating_illegal() {
        use sim::Sim;
        let illegals = [0x00u8, 0x0A, 0x0D, 0x22, 0x26, 0x5C];
        for &il in &illegals {
            for &cl in &[0x41u8, 0x7F, 0x61] {
                for len in [14usize, 15, 16, 28, 29, 30, 42, 56] {
                    let data: Vec<u8> = (0..len).map(|i| if i % 2 == 0 { il } else { cl }).collect();
                    let mut out = Vec::with_capacity(enc_cap(len));
                    encode_with::<Sim>(&data, &mut out);
                    let ref_enc = encode_ref(&data);
                    assert_eq!(out, ref_enc, "encode mismatch il={:02x} cl={:02x} len={}", il, cl, len);
                    let mut dec = Vec::with_capacity(dec_cap(out.len()));
                    let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
                    unsafe { dec.set_len(n) };
                    assert_eq!(dec, data, "roundtrip mismatch il={:02x} cl={:02x} len={}", il, cl, len);
                }
            }
        }
    }

    /// Fuzz-test: many random blocks of random lengths, comparing SIM
    /// codec against the reference for both encode and decode.
    #[test]
    fn test_sim_fuzz_large() {
        use rand::Rng;
        use rand::RngExt;
        use sim::Sim;
        let mut rng = rand::rng();
        for _ in 0..5000 {
            let len = rng.random_range(1..4097usize);
            let mut data = vec![0u8; len];
            rng.fill_bytes(&mut data);
            let mut out = Vec::with_capacity(enc_cap(len));
            encode_with::<Sim>(&data, &mut out);
            let ref_enc = encode_ref(&data);
            assert_eq!(out, ref_enc, "encode mismatch len={}", len);
            let mut dec = Vec::with_capacity(dec_cap(out.len()));
            let n = decode_with::<Sim>(&out, &mut dec).expect("decode failed");
            unsafe { dec.set_len(n) };
            assert_eq!(dec, data, "roundtrip mismatch len={}", len);
        }
    }

    /// Same byte pattern encoded/decoded at both block sizes (14 and 28
    /// bytes), cross-checked against the scalar reference.
    #[test]
    fn test_sim_encode_block_direct() {
        use sim::Sim;
        for len in [14usize, 28] {
            let data = vec![0x03u8; len];
            let mut enc_sim = Vec::with_capacity(enc_cap(len));
            encode_with::<Sim>(&data, &mut enc_sim);
            let ref_enc = encode_ref(&data);
            assert_eq!(enc_sim, ref_enc, "encode mismatch len={}", len);

            // Both the scalar and the SIM decoder must round-trip the data.
            let dec_scalar = decode_ref(&enc_sim).expect("scalar decode of sim enc");
            assert_eq!(dec_scalar, data, "scalar decode of sim enc len={}", len);
            assert_eq!(
                decode_ref(&ref_enc).expect("scalar decode of ref enc"),
                data,
                "scalar decode of ref enc len={}",
                len
            );

            let mut dec_sim = Vec::with_capacity(dec_cap(enc_sim.len()));
            let n = decode_with::<Sim>(&enc_sim, &mut dec_sim).expect("sim decode");
            unsafe { dec_sim.set_len(n) };
            if dec_sim != data {
                panic!(
                    "SIM decode mismatch len={}\n  expected: {:?}\n  got:      {:?}",
                    len, data, dec_sim
                );
            }
        }
    }

    /// Verify that the vector decode correctly rejects invalid streams
    /// for a wide variety of corruptions.
    #[test]
    fn test_sim_decode_invalid_extensive() {
        use rand::Rng;
        use rand::RngExt;
        use sim::Sim;
        let mut rng = rand::rng();
        for _ in 0..3000 {
            let len = rng.random_range(2..258usize);
            let mut data = vec![0u8; len];
            rng.fill_bytes(&mut data);
            // Force some high bytes to create invalid but plausible streams.
            for b in data.iter_mut() {
                if rng.random_range(0..3u32) == 0 {
                    *b |= 0x80;
                }
            }
            let mut out = Vec::with_capacity(enc_cap(len));
            encode_with::<Sim>(&data, &mut out);
            let sim_result = decode_with_vec::<Sim>(&out).ok();
            let ref_result = decode_ref(&out).ok();
            match (sim_result, ref_result) {
                (None, None) => {},
                (Some(a), Some(b)) => assert_eq!(a, b, "decode mismatch on {:?}", out),
                _ => panic!("decode outcome mismatch on len={}", len),
            }
        }
    }
