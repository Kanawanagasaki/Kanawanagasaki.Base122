//! AVX2-accelerated encode/decode kernels, compiled when the build target
//! has AVX2 (`cfg(target_feature = "avx2")`, e.g. `-C target-cpu=x86-64-v3`).
//!
//! The 7-bit group expansion (encode) and packing (decode) are bit
//! permutations; without `pext`/`pdep` they are done with `vpsrlvq` (per-lane
//! variable shifts) plus `vpshufb` byte collection. Illegal-byte detection uses
//! two `vpshufb` bit-matrix lookups (low/high nibble), and escapes are spliced
//! with `vpblendvb`.
//!
//! ## Encode
//!
//! 28 input bytes = 32 groups. Blocks without escapes take a straight store.
//! Dirty blocks compute the escape pair bytes for every position in one batched
//! pass: the hit mask is looked up directly to the pair lead `b1` (`0xC2 |
//! ((class-1) << 2)`), whose bit 7 doubles as the blend mask, plus bit 6 of the
//! next group; the continuation `b2` is looked up from the unshifted groups and
//! placed one byte right by a shifted blend mask — the two blend masks do the
//! splicing, so only one cross-position byte shift (`next_arr`) is needed.
//! `vpshufb` shifts stay within 128-bit halves; the single cross-half byte
//! (position 15, whose "next" lives in the other half) selects a slow path
//! with `vpalignr`-based shifts. Adjacent illegal groups fall back to the
//! scalar codec (their escapes would overlap).
//!
//! ## Decode
//!
//! Every 8 stream bytes are exactly 8 groups (a group byte is one group, an
//! escape pair is two), so each 8-byte block decodes to exactly 7 output bytes
//! and blocks are independent and byte-aligned. A 32-byte window (4 blocks) is
//! transformed byte-wise: group bytes pass through, leads become their illegal
//! byte, continuations become `(lead & 1) << 6 | (cont & 0x3F)`. The 8 groups
//! of each block are bit-gathered with two scalar `pext`s (BMI2). Validity is
//! checked vectorized (lead/continuation structure); any anomaly hands the rest
//! to the scalar codec, which produces the exact error. A pair straddling the
//! end of the vector region is completed by the caller with the last lead byte.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]
use core::arch::x86_64::*;

// --- Encode tables ---------------------------------------------------------

/// Bit-matrix tables: `LOW[v & 0xF] & HIGH[v >> 4] != 0` iff `v` is one of the
/// six illegal values (0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C).
const LOW_TABLE: [u8; 16] = [0x01, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x20, 0x04, 0x00, 0x00];
// Values 0x00, 0x0A, 0x0D all have high nibble 0, so H[0] carries bits 0-2.
const HIGH_TABLE: [u8; 16] = [0x07, 0x00, 0x18, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Escape lead base byte, looked up directly from the hit mask (bit flags
/// 1,2,4,8,16,32 for the six illegal values): `B1BASE_L[hit & 0xF] |
/// B1BASE_H[hit >> 4]` = `0xC2 | ((class-1) << 2)`, elsewhere 0. Non-zero at
/// an illegal position, and bit 7 is always set there, so the same value
/// doubles as the blend mask.
const B1BASE_L: [u8; 16] = [0x00, 0xC2, 0xC6, 0x00, 0xCA, 0x00, 0x00, 0x00, 0xCE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
const B1BASE_H: [u8; 16] = [0x00, 0xD2, 0xD6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Within-half byte shifts used on the fast dirty path. Byte 15 of a left
/// shift and byte 16 of a right shift cross the 128-bit halves; those
/// positions are only consumed when group 15 is illegal (slow path) or group
/// 31 is illegal (pending path), both excluded here.
const SL_IDX: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0];
/// Right shift of the fold mask: byte 16 is zeroed (it should be f15, which is
/// the slow-path trigger), so the continuation of a pair at 15 is never
/// written by the fast path.
const SR_CF_IDX: [u8; 32] = [
    0xFF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    0xFF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
];

/// Reverse the bytes of an 8-byte window (bytes 0-6 become the BE 56-bit
/// value; byte 7 is the overread byte, harmless).
const REV8: [u8; 16] = [6, 5, 4, 3, 2, 1, 0, 7, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
/// Reverse the bytes within each 64-bit lane (replaces two scalar `bswap`s).
const REV64_LANE: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// Collect lane 0's and lane 1's byte 0 into output bytes 0, 1 of each half.
const COLLECT: [u8; 16] = [0, 8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

/// Like COLLECT but into output bytes 4, 5 of each half.
const COLLECT2: [u8; 16] = [0xFF, 0xFF, 0xFF, 0xFF, 0, 8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

/// Compact one round's dword-rearranged groups into the low half's bytes 0-7.
const COMPACT: [u8; 16] = [0, 1, 4, 5, 8, 9, 12, 13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

// --- Decode tables ---------------------------------------------------------

/// `ILLEGAL_T[(b >> 2) & 0xF]` = the illegal byte for a lead byte's class
/// index (0-5; 6/7 are the shortened marker, excluded before storing).
const ILLEGAL_T: [u8; 16] = [0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Byte j = src[j-1] over the original window: byte 0 and byte 16 are zeroed
/// (indices 0xFF) and patched by the caller afterwards with the previous
/// block's last byte and this block's byte 15 respectively.
const D_SR_IDX: [u8; 32] = [
    0xFF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    0xFF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
];

/// pext mask: gather the low 7 bits of each of the 8 group bytes into the
/// 56-bit output stream (group j at bits 7j..7j+6), stored as 8 LE bytes.
const LO7: u64 = crate::scalar::LO7_8;

#[inline(always)]
unsafe fn load128(t: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(t.as_ptr() as *const __m128i) }
}

#[inline(always)]
unsafe fn load256(t: &[u8; 32]) -> __m256i {
    unsafe { _mm256_loadu_si256(t.as_ptr() as *const __m256i) }
}

/// Expand one 56-bit BE value (in bytes 0-6 of `q`) into 8 groups, laid out
/// at bytes 0,1,16,17,4,5,20,21 of the result (see FINALIZE).
#[inline(always)]
unsafe fn expand_one(
    q: __m128i,
    rev: __m128i,
    count_lo: __m256i,
    count_hi: __m256i,
    coll: __m256i,
    coll2: __m256i,
    sevenf: __m256i,
) -> __m256i {
    let r = unsafe { _mm_shuffle_epi8(q, rev) };
    let x = unsafe { _mm256_broadcastq_epi64(r) };
    let a = unsafe { _mm256_shuffle_epi8(_mm256_srlv_epi64(x, count_lo), coll) };
    let b = unsafe { _mm256_shuffle_epi8(_mm256_srlv_epi64(x, count_hi), coll2) };
    unsafe { _mm256_and_si256(_mm256_or_si256(a, b), sevenf) }
}

/// Shift a 32-byte value left by one byte across the 128-bit halves:
/// byte k = src[k+1], byte 15 = the high half's byte 0.
#[inline(always)]
unsafe fn cross_shift_l(v: __m256i) -> __m256i {
    let lo = unsafe { _mm256_castsi256_si128(v) };
    let hi = unsafe { _mm256_extracti128_si256(v, 1) };
    let lo_l = unsafe { _mm_alignr_epi8(hi, lo, 1) };
    let hi_l = unsafe { _mm_alignr_epi8(hi, hi, 1) };
    unsafe { _mm256_inserti128_si256(_mm256_castsi128_si256(lo_l), hi_l, 1) }
}

/// Shift a 32-byte value right by one byte across the 128-bit halves:
/// byte k = src[k-1], byte 0 = 0.
#[inline(always)]
unsafe fn cross_shift_r(v: __m256i) -> __m256i {
    let lo = unsafe { _mm256_castsi256_si128(v) };
    let hi = unsafe { _mm256_extracti128_si256(v, 1) };
    let zero = unsafe { _mm_setzero_si128() };
    let lo_r = unsafe { _mm_alignr_epi8(lo, zero, 15) };
    let hi_r = unsafe { _mm_alignr_epi8(hi, lo, 15) };
    unsafe { _mm256_inserti128_si256(_mm256_castsi128_si256(lo_r), hi_r, 1) }
}

#[inline(always)]
unsafe fn byte_at(v: __m256i, p: usize) -> u8 {
    // Runtime-position extraction: shuffle byte `p` into byte 0.
    let half = if p < 16 {
        unsafe { _mm256_castsi256_si128(v) }
    } else {
        unsafe { _mm256_extracti128_si256(v, 1) }
    };
    let idx = unsafe { _mm_set1_epi8((p & 15) as i8) };
    let s = unsafe { _mm_shuffle_epi8(half, idx) };
    unsafe { _mm_cvtsi128_si32(s) as u8 }
}

/// Detect AVX2 and BMI2 via CPUID (no_std-safe).
pub(crate) fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let res = unsafe { core::arch::x86_64::__cpuid_count(7, 0) };
        (res.ebx & ((1 << 5) | (1 << 8))) == ((1 << 5) | (1 << 8))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX2 encode: processes 28-byte blocks (32 groups each) until a block needs
/// the scalar pending mechanism or the adjacent-illegal pairing, then hands
/// control back to the scalar loop.
///
/// # Safety
/// `out_ptr` must have capacity for the full output; `data` is overread by up
/// to 4 bytes (same convention as the scalar fast path).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_28_avx2(
    data: &[u8],
    out_ptr: *mut u8,
    out_pos: &mut usize,
    i: &mut usize,
    pending_illegal_bits: &mut u8,
    has_pending_illegal: &mut bool,
) {
    let len = data.len();
    let ptr = data.as_ptr();

    let rev = unsafe { load128(&REV8) };
    let coll = unsafe { _mm256_broadcastsi128_si256(load128(&COLLECT)) };
    let coll2 = unsafe { _mm256_broadcastsi128_si256(load128(&COLLECT2)) };
    let compact = unsafe { _mm256_broadcastsi128_si256(load128(&COMPACT)) };
    let dperm_round = unsafe { _mm256_set_epi32(-1, -1, -1, -1, 5, 1, 4, 0) };
    let dperm_lo = unsafe { _mm256_set_epi32(-1, -1, -1, -1, 5, 4, 1, 0) };
    let dperm_hi = unsafe { _mm256_set_epi32(5, 4, 1, 0, -1, -1, -1, -1) };
    let foldcls_l_p = unsafe { _mm256_broadcastsi128_si256(load128(&B1BASE_L)) };
    let foldcls_h_p = unsafe { _mm256_broadcastsi128_si256(load128(&B1BASE_H)) };
    let sl_p = unsafe { _mm256_broadcastsi128_si256(load128(&SL_IDX)) };
    let srcf_p = unsafe { load256(&SR_CF_IDX) };
    let one_mask = unsafe { _mm256_set1_epi8(0x01) };
    let eighty = unsafe { _mm256_set1_epi8(0x80u8 as i8) };
    let t3f = unsafe { _mm256_set1_epi8(0x3F) };
    let clear31 = unsafe {
        // set_epi8(e31..e0): 0x7F must be the FIRST argument (byte 31).
        _mm256_set_epi8(
            0x7F, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        )
    };
    let low_t = unsafe { load128(&LOW_TABLE) };
    let high_t = unsafe { load128(&HIGH_TABLE) };
    let low_p = unsafe { _mm256_broadcastsi128_si256(low_t) };
    let high_p = unsafe { _mm256_broadcastsi128_si256(high_t) };
    let sevenf = unsafe { _mm256_set1_epi8(0x7F) };
    let zero4 = unsafe { _mm256_set1_epi8(0x0F) };
    let count_lo = unsafe { _mm256_set_epi64x(28, 35, 42, 49) };
    let count_hi = unsafe { _mm256_set_epi64x(0, 7, 14, 21) };

    while *i + 29 <= len {
        if *has_pending_illegal {
            break;
        }

        let q0 = unsafe { _mm_loadl_epi64(ptr.add(*i) as *const __m128i) };
        let q1 = unsafe { _mm_loadl_epi64(ptr.add(*i + 7) as *const __m128i) };
        let q2 = unsafe { _mm_loadl_epi64(ptr.add(*i + 14) as *const __m128i) };
        let q3 = unsafe { _mm_loadl_epi64(ptr.add(*i + 21) as *const __m128i) };

        let g0 = unsafe { expand_one(q0, rev, count_lo, count_hi, coll, coll2, sevenf) };
        let g1 = unsafe { expand_one(q1, rev, count_lo, count_hi, coll, coll2, sevenf) };
        let g2 = unsafe { expand_one(q2, rev, count_lo, count_hi, coll, coll2, sevenf) };
        let g3 = unsafe { expand_one(q3, rev, count_lo, count_hi, coll, coll2, sevenf) };

        // Per window: the raw expansion has its four group pairs at dwords
        // 0,2,4,6 (order 01,45,23,67); vpshufb cannot reach across halves, so
        // use a dword-level vpermd first, then a half-local compact shuffle.
        let h0 = unsafe { _mm256_permutevar8x32_epi32(g0, dperm_round) };
        let h1 = unsafe { _mm256_permutevar8x32_epi32(g1, dperm_round) };
        let h2 = unsafe { _mm256_permutevar8x32_epi32(g2, dperm_round) };
        let h3 = unsafe { _mm256_permutevar8x32_epi32(g3, dperm_round) };
        let f0 = unsafe { _mm256_shuffle_epi8(h0, compact) };
        let f1 = unsafe { _mm256_shuffle_epi8(h1, compact) };
        let f2 = unsafe { _mm256_shuffle_epi8(h2, compact) };
        let f3 = unsafe { _mm256_shuffle_epi8(h3, compact) };
        let p01 = unsafe { _mm256_permute2x128_si256(f0, f1, 0x20) };
        let p23 = unsafe { _mm256_permute2x128_si256(f2, f3, 0x20) };
        let groups = unsafe {
            _mm256_or_si256(
                _mm256_permutevar8x32_epi32(p01, dperm_lo),
                _mm256_permutevar8x32_epi32(p23, dperm_hi),
            )
        };

        let lo = unsafe { _mm256_shuffle_epi8(low_p, groups) };
        // srli_epi16 leaks the neighbour's low bits into bits 4-7 of the index
        // byte; mask to 4 bits so bit 7 stays clear (vpshufb zeroes it).
        let hi = unsafe {
            _mm256_shuffle_epi8(high_p, _mm256_and_si256(_mm256_srli_epi16(groups, 4), zero4))
        };
        let hit = unsafe { _mm256_and_si256(lo, hi) };

        // Branch-free dirty transform: for escape-free blocks b1base is
        // all-zero, so the blend masks are empty and both blends are no-ops —
        // the stored result is `groups` unchanged. Running the escape
        // machinery unconditionally avoids a ~50%-mispredicting dispatch
        // branch on random data (the escapes are ~53% of blocks). The hit
        // mask is looked up directly to the pair lead byte, whose bit 7
        // doubles as the blend mask (no separate fold/class pass).
        let b1base = unsafe {
            _mm256_or_si256(
                _mm256_shuffle_epi8(foldcls_l_p, hit),
                _mm256_shuffle_epi8(foldcls_h_p, _mm256_and_si256(_mm256_srli_epi16(hit, 4), zero4)),
            )
        };
        let m = unsafe { _mm256_movemask_epi8(b1base) } as u32;

        // Adjacent illegal groups (each escape consumes its successor) need
        // the scalar pairing logic; they are rare (~7% of dirty blocks).
        if (m & (m >> 1)) != 0 {
            break;
        }

        let (next_arr, cf) = if (m & 0x8000) != 0 {
            // Group 15's next group lives across the 128-bit half boundary:
            // use vpalignr shifts that cross halves.
            (
                unsafe { cross_shift_l(groups) },
                unsafe { cross_shift_r(b1base) },
            )
        } else {
            (unsafe { _mm256_shuffle_epi8(groups, sl_p) }, unsafe {
                _mm256_shuffle_epi8(b1base, srcf_p)
            })
        };
        let nb = unsafe { _mm256_and_si256(_mm256_srli_epi16(next_arr, 6), one_mask) };
        let b1 = unsafe { _mm256_or_si256(b1base, nb) };
        let b2 = unsafe { _mm256_or_si256(_mm256_and_si256(groups, t3f), eighty) };
        // clear31 keeps byte 31 intact (a group value) so the pending check
        // below reads the raw illegal byte; pending blocks may contain other
        // escapes, so they are spliced like any other block.
        let groups = unsafe {
            _mm256_blendv_epi8(groups, b1, _mm256_and_si256(b1base, clear31))
        };
        let groups = unsafe { _mm256_blendv_epi8(groups, b2, cf) };

        if (m & 0x8000_0000) != 0 {
            // Escape at the block's last group: defer via the pending
            // mechanism and hand back to the scalar path. The stored byte at
            // the escape position is garbage and will be overwritten by the
            // pending pair.
            let bp = unsafe { byte_at(groups, 31) };
            *pending_illegal_bits = bp;
            *has_pending_illegal = true;
            unsafe { _mm256_storeu_si256(out_ptr.add(*out_pos) as *mut __m256i, groups) };
            *out_pos += 31;
            *i += 28;
            break;
        }

        unsafe { _mm256_storeu_si256(out_ptr.add(*out_pos) as *mut __m256i, groups) };
        *out_pos += 32;
        *i += 28;
        continue;
    }
}


/// AVX2 decode: processes 32-byte windows (4 blocks of 8 stream bytes) until
/// an anomaly requires the scalar codec. Returns the number of input bytes
/// consumed (a multiple of 32) and whether the last consumed byte was an
/// escape lead (its continuation is the first unconsumed byte).
///
/// # Safety
/// `ptr` must point to `len` readable bytes; `out_ptr` must have capacity for
/// `len * 7 / 8 + 8` bytes. The caller must have verified AVX2+BMI2 support.
#[target_feature(enable = "avx2,bmi2")]
pub(crate) unsafe fn decode_blocks_avx2(
    ptr: *const u8,
    len: usize,
    out_ptr: *mut u8,
) -> (usize, bool, bool) {
    let i80 = unsafe { _mm256_set1_epi8(0x80u8 as i8) };
    let c0 = unsafe { _mm256_set1_epi8(0xC0u8 as i8) };
    // valid lead = (b & 0xE2) == 0xC2: bits 7-6 = 11, bit 5 = 0, bit 1 = 1.
    // (0xC2 alone also matches 0xE0-0xFF, which are invalid leads.)
    let c2 = unsafe { _mm256_set1_epi8(0xE2u8 as i8) };
    let c2v = unsafe { _mm256_set1_epi8(0xC2u8 as i8) };
    let fe = unsafe { _mm256_set1_epi8(0xFEu8 as i8) };
    let de = unsafe { _mm256_set1_epi8(0xDEu8 as i8) };
    let t3f = unsafe { _mm256_set1_epi8(0x3F) };
    // bit 6 of each byte: 0x40 in the low byte of a 16-bit lane, 0x4000 in
    // the high byte (the slli16 shift crosses into the high byte).
    let t40 = unsafe { _mm256_set1_epi16(0x4040) };
    let zero4 = unsafe { _mm256_set1_epi8(0x0F) };
    let illegal_p = unsafe { _mm256_broadcastsi128_si256(load128(&ILLEGAL_T)) };
    let dsr_p = unsafe { load256(&D_SR_IDX) };
    let rev64 = unsafe { load128(&REV64_LANE) };
    let clr31 = unsafe {
        // set_epi8(e31..e0): 0x00 must be the FIRST argument (byte 31).
        _mm256_set_epi8(
            0x00, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        )
    };
    // clear byte 0: the straddle's continuation is validated in scalar code.
    let clr0 = unsafe {
        _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x00,
        )
    };

    let mut i = 0usize;
    let mut out_pos = 0usize;
    let mut bad = false;

    // Keep the last 64 bytes for the scalar tail: the stream's final escape
    // pair may be a shortened marker (0xDE/0xDF), which the encoder only
    // emits at the very end and which the vector path cannot process.
    while i + 64 <= len {
        let b = unsafe { _mm256_loadu_si256(ptr.add(i) as *const __m256i) };

        // Masks: lead/continuation candidates.
        let bc0 = unsafe { _mm256_and_si256(b, c0) };
        let lead = unsafe { _mm256_cmpeq_epi8(bc0, c0) };
        let cont = unsafe { _mm256_cmpeq_epi8(bc0, i80) };

        // Transform: groups pass through, leads become their illegal byte,
        // continuations become (lead & 1) << 6 | (b & 0x3F). The shift is a
        // per-128-bit-lane byte shift, so bytes 0 and 16 are zeroed; their
        // missing (lead & 1) bit is patched into the packed output scalar-wise
        // (bit 6 of group 0 / group 16 = bit 55 of v0 / v2).
        let next_src = unsafe { _mm256_shuffle_epi8(b, dsr_p) };
        let next = unsafe {
            _mm256_or_si256(
                _mm256_and_si256(_mm256_slli_epi16(next_src, 6), t40),
                _mm256_and_si256(b, t3f),
            )
        };
        let illegal = unsafe {
            _mm256_shuffle_epi8(illegal_p, _mm256_and_si256(_mm256_srli_epi16(b, 2), zero4))
        };
        let g = unsafe { _mm256_blendv_epi8(b, illegal, lead) };
        let g = unsafe { _mm256_blendv_epi8(g, next, cont) };

        // Pack: each block's 8 groups -> the 56-bit output stream -> 7 bytes.
        // Byte-reverse both 64-bit lanes with one vpshufb (cheaper than two
        // scalar bswaps), then pext each lane.
        let g0 = unsafe { _mm256_castsi256_si128(g) };
        let g1 = unsafe { _mm256_extracti128_si256(g, 1) };
        let g0r = unsafe { _mm_shuffle_epi8(g0, rev64) };
        let g1r = unsafe { _mm_shuffle_epi8(g1, rev64) };
        let b0 = unsafe { _mm_cvtsi128_si64(g0r) } as u64;
        let b1 = unsafe { _mm_extract_epi64(g0r, 1) } as u64;
        let b2 = unsafe { _mm_cvtsi128_si64(g1r) } as u64;
        let b3 = unsafe { _mm_extract_epi64(g1r, 1) } as u64;
        let mut v0 = unsafe { _pext_u64(b0, LO7) };
        let v1 = unsafe { _pext_u64(b1, LO7) };
        let mut v2 = unsafe { _pext_u64(b2, LO7) };
        let v3 = unsafe { _pext_u64(b3, LO7) };

        // Straddle patches: byte 0's continuation lead is the previous
        // block's last byte, byte 16's is this block's byte 15.
        let prev = if i > 0 { unsafe { *ptr.add(i - 1) } } else { 0 };
        let patch0 = ((unsafe { *ptr.add(i) } & 0xC0) == 0x80 && (prev & 1) != 0) as u64;
        let patch16 = ((unsafe { *ptr.add(i + 16) } & 0xC0) == 0x80
            && (unsafe { *ptr.add(i + 15) } & 1) != 0) as u64;
        v0 |= patch0 << 55;
        v2 |= patch16 << 55;
        let op = unsafe { out_ptr.add(out_pos) };
        unsafe {
            (op as *mut u64).write_unaligned((v0 << 8).to_be());
            (op.add(7) as *mut u64).write_unaligned((v1 << 8).to_be());
            (op.add(14) as *mut u64).write_unaligned((v2 << 8).to_be());
            (op.add(21) as *mut u64).write_unaligned((v3 << 8).to_be());
        }

        // Structure errors: a lead must be valid, every lead must be followed
        // by a continuation, every continuation must follow a lead. Byte 0's
        // continuation partner is the previous block's last byte and byte 31's
        // partner is byte 32 (outside the window); both are checked in scalar
        // code (byte 31 by the caller's prelude). The checks run in parallel
        // with the transform and only set a flag: the loop is branch-free, and
        // an anomalous block makes the caller re-decode with the scalar codec
        // (which overwrites this iteration's speculative stores).
        let valid = unsafe { _mm256_cmpeq_epi8(_mm256_and_si256(b, c2), c2v) };
        let short = unsafe { _mm256_cmpeq_epi8(_mm256_and_si256(b, fe), de) };
        let err_inv = unsafe { _mm256_andnot_si256(valid, lead) };
        let sr_lead = unsafe { cross_shift_r(lead) };
        // Byte 0 of err_cont is legitimately set for a straddling pair (its
        // lead is the previous block's last byte, checked in scalar code).
        let err_cont = unsafe {
            _mm256_and_si256(_mm256_andnot_si256(sr_lead, cont), clr0)
        };
        let sr_cont = unsafe { cross_shift_l(cont) };
        let err_pair = unsafe {
            _mm256_and_si256(_mm256_andnot_si256(sr_cont, lead), clr31)
        };
        let err = unsafe {
            _mm256_or_si256(
                _mm256_or_si256(err_inv, err_cont),
                _mm256_or_si256(err_pair, short),
            )
        };
        bad |= unsafe { _mm256_testz_si256(err, err) } == 0;
        bad |= (unsafe { *ptr.add(i) } & 0xC0) == 0x80 && (prev & 0xE2) != 0xC2;

        i += 32;
        out_pos += 28;
    }

    let last_lead = i > 0 && unsafe { *ptr.add(i - 1) } & 0xE2 == 0xC2;
    (i, last_lead, bad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{load56_be_overread1, split56_to_groups_le};
    use alloc::{vec, vec::Vec};
    use crate::String;

    fn expand_buf(buf: &[u8]) -> [u8; 32] {
        unsafe {
            let rev = _mm_loadu_si128(REV8.as_ptr() as *const __m128i);
            let coll = _mm256_broadcastsi128_si256(_mm_loadu_si128(COLLECT.as_ptr() as *const __m128i));
            let coll2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(COLLECT2.as_ptr() as *const __m128i));
            let compact = _mm256_broadcastsi128_si256(_mm_loadu_si128(COMPACT.as_ptr() as *const __m128i));
            let sevenf = _mm256_set1_epi8(0x7F);
            let count_lo = _mm256_set_epi64x(28, 35, 42, 49);
            let count_hi = _mm256_set_epi64x(0, 7, 14, 21);
            let dperm_round = _mm256_set_epi32(-1, -1, -1, -1, 5, 1, 4, 0);
            let dperm_lo = _mm256_set_epi32(-1, -1, -1, -1, 5, 4, 1, 0);
            let dperm_hi = _mm256_set_epi32(5, 4, 1, 0, -1, -1, -1, -1);
            let q0 = _mm_loadl_epi64(buf.as_ptr() as *const __m128i);
            let q1 = _mm_loadl_epi64(buf.as_ptr().add(7) as *const __m128i);
            let q2 = _mm_loadl_epi64(buf.as_ptr().add(14) as *const __m128i);
            let q3 = _mm_loadl_epi64(buf.as_ptr().add(21) as *const __m128i);
            let g0 = expand_one(q0, rev, count_lo, count_hi, coll, coll2, sevenf);
            let g1 = expand_one(q1, rev, count_lo, count_hi, coll, coll2, sevenf);
            let g2 = expand_one(q2, rev, count_lo, count_hi, coll, coll2, sevenf);
            let g3 = expand_one(q3, rev, count_lo, count_hi, coll, coll2, sevenf);
            let h0 = _mm256_permutevar8x32_epi32(g0, dperm_round);
            let h1 = _mm256_permutevar8x32_epi32(g1, dperm_round);
            let h2 = _mm256_permutevar8x32_epi32(g2, dperm_round);
            let h3 = _mm256_permutevar8x32_epi32(g3, dperm_round);
            let f0 = _mm256_shuffle_epi8(h0, compact);
            let f1 = _mm256_shuffle_epi8(h1, compact);
            let f2 = _mm256_shuffle_epi8(h2, compact);
            let f3 = _mm256_shuffle_epi8(h3, compact);
            let p01 = _mm256_permute2x128_si256(f0, f1, 0x20);
            let p23 = _mm256_permute2x128_si256(f2, f3, 0x20);
            let groups = _mm256_or_si256(
                _mm256_permutevar8x32_epi32(p01, dperm_lo),
                _mm256_permutevar8x32_epi32(p23, dperm_hi),
            );
            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, groups);
            out
        }
    }

    #[test]
    fn test_expand_matches_scalar() {
        // targeted: leading small values
        for buf in [
            [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [0x41; 32],
            [0xFF; 32],
        ] {
            let got = expand_buf(&buf);
            let mut want = [0u8; 32];
            for k in 0..4 {
                let bits = unsafe { load56_be_overread1(buf.as_ptr().add(k * 7)) };
                want[k * 8..k * 8 + 8].copy_from_slice(&split56_to_groups_le(bits).to_le_bytes());
            }
            assert_eq!(got, want, "targeted");
        }

        let mut x = 0x1234_5678_9abc_def0u64;
        for seed in 0..2000 {
            let mut buf = [0u8; 32];
            for b in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
                *b = (x >> 33) as u8;
            }
            let got = expand_buf(&buf);
            let mut want = [0u8; 32];
            for k in 0..4 {
                let bits = unsafe { load56_be_overread1(buf.as_ptr().add(k * 7)) };
                want[k * 8..k * 8 + 8].copy_from_slice(&split56_to_groups_le(bits).to_le_bytes());
            }
            assert_eq!(got, want, "seed {}", seed);
        }
    }

    #[test]
    fn test_membership() {
        unsafe {
            let low_p = _mm256_broadcastsi128_si256(_mm_loadu_si128(LOW_TABLE.as_ptr() as *const __m128i));
            let high_p = _mm256_broadcastsi128_si256(_mm_loadu_si128(HIGH_TABLE.as_ptr() as *const __m128i));
            for v in 0..128u8 {
                let bytes = [v; 32];
                let g = _mm256_loadu_si256(bytes.as_ptr() as *const __m256i);
                let lo = _mm256_shuffle_epi8(low_p, g);
                let hi = _mm256_shuffle_epi8(high_p, _mm256_and_si256(_mm256_srli_epi16(g, 4), _mm256_set1_epi8(0x0F)));
                let hit = _mm256_and_si256(lo, hi);
                let is_illegal = [0u8, 10, 13, 34, 38, 92].contains(&v);
                assert_eq!(_mm256_testz_si256(hit, hit) != 0, !is_illegal, "v={}", v);
            }
        }
    }

    /// Reference encoder for one 28-byte block, mirroring the scalar codec's
    /// group walk (including adjacent escapes); the pending case (group 31
    /// illegal) is out of scope for the AVX2 block itself.
    fn reference_encode_28(bytes: &[u8]) -> Vec<u8> {
        let mut groups = [0u8; 32];
        for k in 0..4 {
            let bits = unsafe { load56_be_overread1(bytes.as_ptr().add(k * 7)) };
            let g = split56_to_groups_le(bits).to_le_bytes();
            groups[k * 8..k * 8 + 8].copy_from_slice(&g);
        }
        let mut out = Vec::new();
        let mut p = 0usize;
        while p < 32 {
            let cur = groups[p];
            let class = crate::scalar::CLASS[cur as usize];
            if class == 0 {
                out.push(cur);
                p += 1;
                continue;
            }
            assert!(p + 1 < 32, "pending escape out of scope");
            let next = groups[p + 1];
            let idx = (class - 1) as usize;
            let b1 = 0b1100_0010 | ((idx as u8) << 2) | (next >> 6);
            let b2 = 0x80 | (next & 0x3F);
            out.push(b1);
            out.push(b2);
            p += 2;
        }
        out
    }

    /// Run the AVX2 encode dirty path on one block and compare with the
    /// reference encoder. Blocks the AVX2 loop defers to the scalar (pending
    /// escape at group 31, or adjacent illegal groups) are skipped — those are
    /// covered by the crate-level roundtrip tests.
    fn check_encode_block(buf: &[u8; 32]) {
        let mut groups = [0u8; 32];
        for k in 0..4 {
            let bits = unsafe { load56_be_overread1(buf.as_ptr().add(k * 7)) };
            groups[k * 8..k * 8 + 8].copy_from_slice(&split56_to_groups_le(bits).to_le_bytes());
        }
        for p in 0..32 {
            if crate::CLASS[groups[p] as usize] != 0
                && (p == 31 || crate::CLASS[groups[p + 1] as usize] != 0)
            {
                return; // pending or adjacent -> scalar fallback
            }
        }
        unsafe {
            let mut out = vec![0u8; 64];
            let mut out_pos = 0usize;
            let mut i = 0usize;
            let mut pending = 0u8;
            let mut has_pending = false;
            encode_28_avx2(
                buf,
                out.as_mut_ptr(),
                &mut out_pos,
                &mut i,
                &mut pending,
                &mut has_pending,
            );
            assert!(!has_pending, "pending escape in encode block test");
            assert_eq!(i, 28);
            out.truncate(out_pos);
            assert_eq!(out, reference_encode_28(buf), "block mismatch");
        }
    }

    #[test]
    fn test_encode_dirty_path_matches_reference() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        // 32 bytes: a 28-byte block plus the 4-byte overread tail.
        let mut buf = [0u8; 32];
        // Seeded blocks, biased towards illegal bytes so most exercise the
        // dirty path, including the slow path (byte 15) and pair boundaries.
        for seed in 0..4000u64 {
            for b in buf[..28].iter_mut() {
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
                *b = ((x >> 33) as u8) & (if seed % 2 == 0 { 0x7F } else { 0xFF });
            }
            // Force the slow-path trigger and other interesting positions.
            if seed % 7 == 0 {
                buf[15] = 0;
            }
            if seed % 11 == 0 {
                buf[0] = 0x5C;
            }
            if seed % 13 == 0 {
                buf[26] = 0x22;
            }
            check_encode_block(&buf);
        }
    }

    /// Reference decoder: plain byte walk with an accumulator, independent of
    /// the AVX2 path.
    fn reference_decode(bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
        let mut out = Vec::new();
        let mut acc = 0u64;
        let mut acc_bits = 0u32;
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            if b < 128 {
                let combined = (acc << 7) | b as u64;
                let total = acc_bits + 7;
                let emitted = total >> 3;
                let nb = total & 7;
                if emitted != 0 {
                    out.push((combined >> nb) as u8);
                }
                acc = combined & ((1u64 << nb).wrapping_sub(1));
                acc_bits = nb;
                i += 1;
                continue;
            }
            let code = crate::scalar::LEAD_DECODE[b as usize];
            if code == 0xFF {
                return Err("Invalid lead byte");
            }
            if i + 1 >= bytes.len() {
                return Err("Unexpected end of input");
            }
            let b2 = bytes[i + 1];
            if (b2 & 0xC0) != 0x80 {
                return Err("Invalid continuation byte");
            }
            i += 2;
            let illegal_index = (code >> 1) as usize;
            let first_bit = code & 1;
            let next = (first_bit << 6) | (b2 & 0x3F);
            let hi = if illegal_index < 6 {
                crate::scalar::ILLEGALS[illegal_index]
            } else {
                // shortened: single group
                let combined = (acc << 7) | next as u64;
                let total = acc_bits + 7;
                let emitted = total >> 3;
                let nb = total & 7;
                if emitted != 0 {
                    out.push((combined >> nb) as u8);
                }
                acc = combined & ((1u64 << nb).wrapping_sub(1));
                acc_bits = nb;
                continue;
            };
            let combined = (acc << 14) | ((hi as u64) << 7) | next as u64;
            let total = acc_bits + 14;
            let emitted = total >> 3;
            let nb = total & 7;
            if emitted != 0 {
                let out_bits = combined >> nb;
                if emitted == 2 {
                    out.push((out_bits >> 8) as u8);
                }
                out.push(out_bits as u8);
            }
            acc = combined & ((1u64 << nb).wrapping_sub(1));
            acc_bits = nb;
        }
        Ok(out)
    }

    #[test]
    fn test_decode_avx2_matches_reference() {
        let mut x = 0x243f_6a88_85a3_08d3u64;
        for seed in 0..2000u64 {
            let len = (seed as usize) % 200 + 1;
            let mut buf = vec![0u8; len];
            for b in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
                *b = (x >> 33) as u8;
            }
            let enc = crate::encode(&buf);
            let want = reference_decode(enc.as_bytes()).unwrap();
            let got = crate::decode(&enc).unwrap();
            assert_eq!(got, want, "seed {} len {}", seed, len);
            assert_eq!(got, buf, "roundtrip seed {} len {}", seed, len);
        }
        // Larger buffer exercising many 32-byte windows and tail crossings.
        for len in [1000usize, 1_000_000] {
            let mut buf = vec![0u8; len];
            for b in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
                *b = (x >> 33) as u8;
            }
            let enc = crate::encode(&buf);
            let want = reference_decode(enc.as_bytes()).unwrap();
            let got = crate::decode(&enc).unwrap();
            assert_eq!(got, want, "len {}", len);
            assert_eq!(got, buf, "roundtrip len {}", len);
        }
    }

    #[test]
    fn test_decode_invalid_agrees_with_reference() {
        let mut x = 0x1319_8a2e_0370_7344u64;
        for seed in 0..5000u64 {
            let len = (seed as usize) % 100 + 2;
            let mut buf = vec![0u8; len];
            for b in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
                *b = (x >> 33) as u8;
            }
            let mut enc = crate::encode(&buf).into_bytes();
            // Corrupt a byte at a random position; the reference decoder
            // decides whether the result is still valid.
            let pos = ((x >> 40) as usize) % enc.len();
            enc[pos] = match seed % 8 {
                0 => 0xC0,
                1 => 0x80,
                2 => 0xDE,
                3 => 0xDF,
                4 => 0xFF,
                5 => 0x00,
                6 => 0x7F,
                _ => 0x40,
            };
            let want = reference_decode(&enc);
            let s = unsafe { String::from_utf8_unchecked(enc) };
            let got = crate::decode(&s);
            assert_eq!(got.is_err(), want.is_err(), "seed {}", seed);
            if let (Ok(g), Ok(w)) = (&got, &want) {
                assert_eq!(g, w, "seed {}", seed);
            }
        }
    }

    /// The AVX2 decode path must agree with the reference decoder on
    /// arbitrary byte strings — most are invalid, and Ok/Err classification
    /// and output must match exactly. Strings are intentionally not valid
    /// UTF-8.
    #[test]
    fn test_decode_fuzz_agrees_with_reference() {
        let mut x = 0x6a09_e667_f3bc_c908u64;
        for seed in 0..5000u64 {
            let len = (seed as usize) % 300 + 1;
            let mut buf = vec![0u8; len];
            for b in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
                *b = (x >> 33) as u8;
            }
            // Bias towards structurally interesting bytes (leads, partial
            // continuations, plain groups).
            for b in buf.iter_mut() {
                match x % 5 {
                    0 => *b = 0xC0 | (*b % 32),
                    1 => *b = 0x80 | (*b % 64),
                    2 => *b %= 128,
                    3 => *b = 0xDF,
                    _ => {}
                }
                x = x.wrapping_mul(6364136223846793407).wrapping_add(1442695040888963407);
            }
            let want = reference_decode(&buf);
            let s = unsafe { String::from_utf8_unchecked(buf) };
            let got = crate::decode(&s);
            assert_eq!(got.is_err(), want.is_err(), "seed {}", seed);
            if let (Ok(g), Ok(w)) = (&got, &want) {
                assert_eq!(g, w, "seed {}", seed);
            }
        }
    }

    /// Every placement of one or two (non-adjacent) illegal values in a
    /// 28-byte block must encode identically to the reference walker. This
    /// exhaustively covers escape pairs at every group boundary, including the
    /// cross-128-bit-half slow path at group 15.
    #[test]
    fn test_encode_exhaustive_illegal_pairs() {
        let il = [0u8, 10, 13, 34, 38, 92];
        let mut buf = [0x41u8; 32];
        // Single illegal at every position.
        for &a in &il {
            for pa in 0..28usize {
                buf[..28].fill(0x41);
                buf[pa] = a;
                check_encode_block(&buf);
            }
        }
        // Pairs of non-adjacent illegals (adjacent pairs fall back to the
        // scalar path inside check_encode_block).
        for &a in &il {
            for &b in &il {
                for pa in 0..28usize {
                    for pb in (pa + 1)..28usize {
                        buf[..28].fill(0x41);
                        buf[pa] = a;
                        buf[pb] = b;
                        check_encode_block(&buf);
                    }
                }
            }
        }
    }

    #[test]
    fn test_encode_seed30_block() {
        // Regression: the exact seed-30 block from test_decode_avx2_matches_reference.
        let buf: [u8; 32] = [
            9, 148, 203, 230, 25, 251, 159, 49, 167, 1, 108, 40, 112, 207, 211, 182,
            170, 95, 81, 57, 198, 198, 251, 201, 47, 109, 64, 113, 7, 83, 169, 0,
        ];
        check_encode_block(&buf);
        // also check the expansion of chunk 3 against split56 directly
        let mut want = [0u8; 32];
        for k in 0..4 {
            let bits = unsafe { load56_be_overread1(buf.as_ptr().add(k * 7)) };
            want[k * 8..k * 8 + 8].copy_from_slice(&split56_to_groups_le(bits).to_le_bytes());
        }
        let got = expand_buf(&buf);
        assert_eq!(got, want, "expand chunk mismatch");
        // invariant: split56(pack(groups)) == groups for the chunk-3 groups
        extern crate std;
        use std::println;
        let g3: [u8; 8] = want[24..32].try_into().unwrap();
        let b3 = u64::from_le_bytes(g3);
        let v3 = unsafe { _pext_u64(b3.swap_bytes(), crate::LO7_8) };
        let back = unsafe { _pdep_u64(v3, crate::LO7_8) }.swap_bytes();
        println!("split56 g3: {:?} pack v3: {:016X} back: {:016X} bytes: {:?}", g3, v3, back, back.to_le_bytes());
    }



}

