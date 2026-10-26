//! NEON-accelerated encode/decode kernels for arm64-v8a.
//!
//! NEON (ASIMD) is a mandatory part of the arm64-v8a architecture, so this
//! module is gated at compile time on `target_arch = "aarch64"` with no
//! runtime detection.
//!
//! ## Encode
//!
//! 14 input bytes = 16 groups. Group `i` of an 8-byte window is
//! `(rev64(v) >> (57 - 7i)) & 0x7F` (the load is byte-reversed so the
//! MSB-first 7-bit windows of the input stream become contiguous bits of the
//! 64-bit value), so four `vshlq_u64` (pairs of 64-bit lanes with shifts
//! -57,-50,-43,-36,-29,-22,-15,-8) produce shifted copies whose byte 0 of
//! each lane holds a complete group, and two `tbl4` gather the 16 groups (one
//! `vext` splices the two halves).
//! Escape splicing mirrors `avx2.rs`: bit-matrix lookups for illegal bytes,
//! the hit mask looked up directly to the `B1BASE` lead byte (bit 7 doubles
//! as the blend mask), arithmetic `b2`, and two byte-wide blends driven by a
//! bit-7-extended mask. There is no cross-half slow path and no movemask: a
//! single `vmaxvq` checks both adjacent-illegal groups and an escape at the
//! block's last group (which would need the scalar pending mechanism).
//!
//! ## Decode
//!
//! Every 8 stream bytes are exactly 8 groups = 56 bits = 7 output bytes, so a
//! 16-byte window (2 blocks) is transformed byte-wise and each block's 8
//! groups are packed into 7 bytes. The pack replaces x86's `pext` with a
//! shift/OR/mask sequence: byte-reverse the 8 group bytes, form 14-bit pairs
//! in 16-bit lanes (`(lane & 0x7F) | ((lane >> 1) & 0x3F80)`), then combine
//! the four pairs from 16-bit to 14-bit spacing with three 64-bit shifts and
//! four masks. Validity is checked vectorized; any anomaly falls back to the
//! scalar codec.
//!
//! ## Testing without ARM hardware
//!
//! The kernels are generic over a `NeonOps` trait: every operation maps to
//! exactly one intrinsic, and a `sim` implementation with identical
//! lane/byte semantics runs on any host, so the full algorithm is verified
//! against the reference codec everywhere (on `aarch64` the intrinsics are
//! additionally cross-checked).

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]

// --- Encode tables (16-byte, index semantics identical to `tbl`) -----------

/// Bit-matrix tables: `LOW[v & 0xF] & HIGH[v >> 4] != 0` iff `v` is one of the
/// six illegal values (0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C).
const LOW_TABLE: [u8; 16] = [0x01, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x20, 0x04, 0x00, 0x00];
const HIGH_TABLE: [u8; 16] = [0x07, 0x00, 0x18, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Escape lead base byte, looked up directly from the hit mask (bit flags
/// 1,2,4,8,16,32 for the six illegal values): `B1BASE_L[hit & 0xF] |
/// B1BASE_H[hit >> 4]` = `0xC2 | ((class-1) << 2)`, elsewhere 0. Non-zero at
/// an illegal position, and bit 7 is always set there, so the same value
/// doubles as the blend mask (no separate fold/class pass).
const B1BASE_L: [u8; 16] = [0x00, 0xC2, 0xC6, 0x00, 0xCA, 0x00, 0x00, 0x00, 0xCE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
const B1BASE_H: [u8; 16] = [0x00, 0xD2, 0xD6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Decode: `ILLEGAL_T[(b >> 2) & 0xF]` = the illegal byte for a lead's class
/// (0-5; 6/7 are the shortened marker, excluded before storing).
const ILLEGAL_T: [u8; 16] = [0x00, 0x0A, 0x0D, 0x22, 0x26, 0x5C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

// --- Shift/index constants --------------------------------------------------

/// 64-bit-lane gather for the byte-reversed layout: group `i` sits at byte 0
/// of 64-bit lane `i % 2` of shifted register `i / 2` (shifts
/// [-57,-50,-43,-36,-29,-22,-15,-8] on the `rev64`-reversed load).
const G_ALL2: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 0, 8, 16, 24, 32, 40, 48, 56];

const fn u64x2(lo: u64, hi: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    let lo_b = lo.to_le_bytes();
    let hi_b = hi.to_le_bytes();
    let mut i = 0;
    while i < 8 {
        b[i] = lo_b[i];
        b[i + 8] = hi_b[i];
        i += 1;
    }
    b
}

/// Pair masks (decode pack): mask low 7 bits of each 16-bit lane's low byte
/// (P7F_LO) or extract the high byte at bits 7-13 (P3F80).
const P7F_LO: [u8; 16] = u64x2(0x007F_007F_007F_007F, 0x007F_007F_007F_007F);
const P3F80: [u8; 16] = u64x2(0x3F80_3F80_3F80_3F80, 0x3F80_3F80_3F80_3F80);

/// Combine masks (decode pack): pairs at 16-bit spacing -> 14-bit spacing.
const M3FFF: [u8; 16] = u64x2(0x3FFF, 0x3FFF);
const M2: [u8; 16] = u64x2(0x0000_0FFF_C000, 0x0000_0FFF_C000);
const M4: [u8; 16] = u64x2(0x03FF_F000_0000, 0x03FF_F000_0000);
const M6: [u8; 16] = u64x2(0x00FF_FC00_0000_0000, 0x00FF_FC00_0000_0000);

const CLR0: [u8; 16] = {
    let mut b = [0xFFu8; 16];
    b[0] = 0x00;
    b
};
const CLR15: [u8; 16] = {
    let mut b = [0xFFu8; 16];
    b[15] = 0x00;
    b
};
const MASK15: [u8; 16] = {
    let mut b = [0x00u8; 16];
    b[15] = 0xFF;
    b
};

// ===========================================================================
// The kernel abstraction: one method per NEON operation, with identical
// semantics across the simulation and the intrinsic implementation.
// ===========================================================================

pub(crate) trait NeonOps: Copy {
    type Reg: Copy;

    /// Load 8 bytes little-endian and duplicate into both 64-bit lanes.
    fn load64_dup(ptr: *const u8) -> Self::Reg;
    /// Load 16 bytes.
    fn load16(ptr: *const u8) -> Self::Reg;
    /// Store 16 bytes.
    fn store16(ptr: *mut u8, r: Self::Reg);
    /// vshlq_u64 with two signed 64-bit counts; negative = right shift.
    fn shlq_u64c(r: Self::Reg, counts: [i64; 2]) -> Self::Reg;
    /// Turn each byte's bit 7 into a full 0xFF/0x00 mask (sign extension).
    fn sshr8(r: Self::Reg) -> Self::Reg;
    /// vqtbl1q_u8: 16-byte table lookup; index bit 7 -> 0.
    fn tbl(table: Self::Reg, idx: Self::Reg) -> Self::Reg;
    /// vqtbl4q_u8: one 64-entry virtual table from four source registers.
    fn tbl4(t0: Self::Reg, t1: Self::Reg, t2: Self::Reg, t3: Self::Reg, idx: Self::Reg) -> Self::Reg;
    /// vextq_u8(a, b, n) = {a[n..16], b[0..n]}.
    fn vext(a: Self::Reg, b: Self::Reg, n: usize) -> Self::Reg;
    /// vbslq_u8: byte = mask bit 7 ? a : b.
    fn bsl(mask: Self::Reg, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// vceqq_u8.
    fn ceq(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    fn band(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    fn bor(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// a & ~b.
    fn bic(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// vshrq_n_u8.
    fn shr8_n(r: Self::Reg, n: u8) -> Self::Reg;
    /// vshlq_n_u8.
    fn shl8_n(r: Self::Reg, n: u8) -> Self::Reg;
    /// vshrq_n_u16.
    fn shr16_n(r: Self::Reg, n: u8) -> Self::Reg;
    /// vshrq_n_u64.
    fn shr64_n(r: Self::Reg, n: u8) -> Self::Reg;
    /// vshlq_n_u64.
    fn shlq_u64(r: Self::Reg, n: u8) -> Self::Reg;
    /// vrev64q_u8: reverse bytes within each 64-bit lane.
    fn rev64(r: Self::Reg) -> Self::Reg;
    /// Splat a byte.
    fn dup_byte(v: u8) -> Self::Reg;
    /// A 16-byte constant.
    fn const16(b: [u8; 16]) -> Self::Reg;
    /// vgetq_lane_u8 (byte j).
/// `get_lane_u8` is only exercised through the Sim implementation by the
/// tests; the intrinsic kernels read lanes via `vgetq_lane_u8` directly.
#[cfg_attr(not(test), allow(dead_code))]
fn get_lane_u8(r: Self::Reg, j: usize) -> u8;
    /// vgetq_lane_u64 (lane j).
    fn get_lane_u64(r: Self::Reg, j: usize) -> u64;
    /// vmaxvq_u8: maximum byte.
    fn maxv(r: Self::Reg) -> u8;
}

// ===========================================================================
// Portable simulation. Each function mirrors the intrinsic with the same
// name and lane/byte semantics; `Reg` is a 128-bit register as 16 bytes.
// Only compiled for tests (the kernels are generic and verified against this
// simulation on any host).
// ===========================================================================


// ===========================================================================
// Generic kernels: one algorithm, verified via `sim`, executed via `Intr`.
// ===========================================================================

/// Encode 14 bytes at `data[i..]` into 16 groups and splice escape pairs.
/// Mirrors `avx2::encode_28_avx2`'s contract: on success advances `i` and
/// `out_pos` and returns true; on an adjacent-illegal block returns false
/// without advancing (the caller falls back to the scalar codec). A pending
/// escape at group 15 stores 15 bytes, sets the pending bits and breaks.
///
/// # Safety
/// `data` must be readable at `*i + 15`; `out_ptr` must have capacity for the
/// full output.
///
/// Only the test suite exercises this 14-byte kernel directly (via `Sim`);
/// production uses the 28-byte [`encode_block_28`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) unsafe fn encode_block<O: NeonOps>(
    data: &[u8],
    out_ptr: *mut u8,
    out_pos: &mut usize,
    i: &mut usize,
    pending_illegal_bits: &mut u8,
    has_pending_illegal: &mut bool,
) -> bool {
    let ptr = data.as_ptr();
    let off = *i;

    // Expand: 2 x 8 bytes -> 16 groups, one per byte. Each 8-byte load is
    // byte-reversed (group i = (rev64(v) >> (57 - 7i)) & 0x7F), then four
    // vshlq_u64 (pairs of 64-bit lanes) place groups at byte 0 of each lane;
    // two tbl4 gather the 16 groups and one vext splices the halves.
    let x_a = O::load64_dup(ptr.add(off));
    let x_b = O::load64_dup(ptr.add(off + 7));
    let r_a = O::rev64(x_a);
    let r_b = O::rev64(x_b);
    let g_all2 = O::const16(G_ALL2);
    let groups = O::band(
        O::vext(
            O::tbl4(
                O::shlq_u64c(r_a, [-57, -50]),
                O::shlq_u64c(r_a, [-43, -36]),
                O::shlq_u64c(r_a, [-29, -22]),
                O::shlq_u64c(r_a, [-15, -8]),
                g_all2,
            ),
            O::tbl4(
                O::shlq_u64c(r_b, [-57, -50]),
                O::shlq_u64c(r_b, [-43, -36]),
                O::shlq_u64c(r_b, [-29, -22]),
                O::shlq_u64c(r_b, [-15, -8]),
                g_all2,
            ),
            8,
        ),
        O::dup_byte(0x80),
    );
    let groups = O::bic(groups, O::dup_byte(0x80));

    // Illegal detection: bit-matrix membership, then look the hit mask up
    // directly to the escape lead byte; its bit 7 doubles as the blend mask
    // (no separate fold/class pass). Escape-free blocks get an all-zero
    // b1base, so the blends below are no-ops and the stored result is
    // `groups` unchanged.
    let low_t = O::const16(LOW_TABLE);
    let high_t = O::const16(HIGH_TABLE);
    let lo = O::tbl(low_t, O::band(groups, O::dup_byte(0x0F)));
    let hi = O::tbl(high_t, O::band(O::shr8_n(groups, 4), O::dup_byte(0x0F)));
    let hit = O::band(lo, hi);
    let fcl = O::const16(B1BASE_L);
    let fch = O::const16(B1BASE_H);
    let b1base = O::bor(
        O::tbl(fcl, hit),
        O::tbl(fch, O::band(O::shr8_n(hit, 4), O::dup_byte(0x0F))),
    );

    // Adjacent illegal groups need the scalar pairing logic; they are rare.
    let adj = O::band(b1base, O::vext(b1base, b1base, 1));
    if O::maxv(adj) != 0 {
        return false;
    }

    // Splice: b1 = b1base | (next >> 6) & 1, b2 = 0x80 | g & 0x3F. The
    // b1base lead bytes double as hit masks only through their bit 7;
    // extend bit 7 to a full-byte mask for the byte-wide vbsl selects.
    let hitm = O::sshr8(b1base);
    let next = O::vext(groups, groups, 1);
    let nb = O::band(O::shr8_n(next, 6), O::dup_byte(0x01));
    let b1 = O::bor(b1base, nb);
    let blend1 = O::bsl(O::band(hitm, O::const16(CLR15)), b1, groups);
    let cf = O::band(O::vext(hitm, hitm, 15), O::const16(CLR0));
    let b2 = O::bor(O::band(groups, O::dup_byte(0x3F)), O::dup_byte(0x80));
    let out = O::bsl(cf, b2, blend1);

    if O::get_lane_u8(b1base, 15) != 0 {
        // Escape at the block's last group: defer via the pending mechanism.
        *pending_illegal_bits = O::get_lane_u8(groups, 15);
        *has_pending_illegal = true;
        O::store16(out_ptr.add(*out_pos), out);
        *out_pos += 15;
        *i += 14;
        return true;
    }

    O::store16(out_ptr.add(*out_pos), out);
    *out_pos += 16;
    *i += 14;
    true
}

/// Encode 28 bytes (two 14-byte blocks) with both blocks interleaved, so the
/// `tbl`/`vext` dependency chains of the two blocks overlap and the per-block
/// fixed costs (constant loads, checks, stores) are amortized. Any anomaly
/// (adjacent illegal groups or an escape at either block's last group) makes
/// the caller fall back to the scalar codec for the whole 28 bytes.
///
/// # Safety
/// `ptr` must have 28 readable bytes at `ptr.add(*i)`; `out_ptr` must have
/// capacity for 32 more bytes.
#[inline(always)]
pub(crate) unsafe fn encode_block_28<O: NeonOps>(
    data: &[u8],
    out_ptr: *mut u8,
    out_pos: &mut usize,
    i: &mut usize,
) -> bool {
    let ptr = data.as_ptr();
    let off = *i;

    // Expand: 4 x 8 bytes -> 32 groups, two blocks interleaved. Each 8-byte
    // load is byte-reversed (group i = (rev64(v) >> (57 - 7i)) & 0x7F), then
    // four vshlq_u64 (pairs of 64-bit lanes) place groups at byte 0 of each
    // lane; two tbl4 gather the 8 groups and one vext splices the halves.
    let x_a = O::load64_dup(ptr.add(off));
    let x_b = O::load64_dup(ptr.add(off + 7));
    let x_c = O::load64_dup(ptr.add(off + 14));
    let x_d = O::load64_dup(ptr.add(off + 21));
    let r_a = O::rev64(x_a);
    let r_b = O::rev64(x_b);
    let r_c = O::rev64(x_c);
    let r_d = O::rev64(x_d);
    let g_all2 = O::const16(G_ALL2);
    let sevenf = O::dup_byte(0x80);
    let g_a = O::bic(
        O::vext(
            O::tbl4(
                O::shlq_u64c(r_a, [-57, -50]),
                O::shlq_u64c(r_a, [-43, -36]),
                O::shlq_u64c(r_a, [-29, -22]),
                O::shlq_u64c(r_a, [-15, -8]),
                g_all2,
            ),
            O::tbl4(
                O::shlq_u64c(r_b, [-57, -50]),
                O::shlq_u64c(r_b, [-43, -36]),
                O::shlq_u64c(r_b, [-29, -22]),
                O::shlq_u64c(r_b, [-15, -8]),
                g_all2,
            ),
            8,
        ),
        sevenf,
    );
    let g_b = O::bic(
        O::vext(
            O::tbl4(
                O::shlq_u64c(r_c, [-57, -50]),
                O::shlq_u64c(r_c, [-43, -36]),
                O::shlq_u64c(r_c, [-29, -22]),
                O::shlq_u64c(r_c, [-15, -8]),
                g_all2,
            ),
            O::tbl4(
                O::shlq_u64c(r_d, [-57, -50]),
                O::shlq_u64c(r_d, [-43, -36]),
                O::shlq_u64c(r_d, [-29, -22]),
                O::shlq_u64c(r_d, [-15, -8]),
                g_all2,
            ),
            8,
        ),
        sevenf,
    );

    // Illegal detection: bit-matrix membership, then the hit mask looked up
    // directly to the escape lead byte (bit 7 doubles as the blend mask).
    let low_t = O::const16(LOW_TABLE);
    let high_t = O::const16(HIGH_TABLE);
    let zero4 = O::dup_byte(0x0F);
    let lo_a = O::tbl(low_t, O::band(g_a, zero4));
    let lo_b = O::tbl(low_t, O::band(g_b, zero4));
    let hi_a = O::tbl(high_t, O::band(O::shr8_n(g_a, 4), zero4));
    let hi_b = O::tbl(high_t, O::band(O::shr8_n(g_b, 4), zero4));
    let hit_a = O::band(lo_a, hi_a);
    let hit_b = O::band(lo_b, hi_b);
    let fcl = O::const16(B1BASE_L);
    let fch = O::const16(B1BASE_H);
    let b1base_a = O::bor(
        O::tbl(fcl, hit_a),
        O::tbl(fch, O::band(O::shr8_n(hit_a, 4), zero4)),
    );
    let b1base_b = O::bor(
        O::tbl(fcl, hit_b),
        O::tbl(fch, O::band(O::shr8_n(hit_b, 4), zero4)),
    );

    // Adjacent illegal groups and an escape at either block's last group
    // (group 15 / group 31, which would need the cross-block pending
    // mechanism) both need the scalar path; they are rare. Folding the
    // pending check into the adjacent mask keeps a single maxv per block
    // instead of the umov/fmov/ccmp chain.
    let m15 = O::const16(MASK15);
    let adj_a = O::bor(
        O::band(b1base_a, O::vext(b1base_a, b1base_a, 1)),
        O::band(b1base_a, m15),
    );
    let adj_b = O::bor(
        O::band(b1base_b, O::vext(b1base_b, b1base_b, 1)),
        O::band(b1base_b, m15),
    );
    // One reduction for both blocks instead of two.
    if O::maxv(O::bor(adj_a, adj_b)) != 0 {
        return false;
    }

    // Splice: b1 = b1base | (next >> 6) & 1, b2 = 0x80 | g & 0x3F. The
    // b1base lead bytes double as hit masks only through their bit 7;
    // extend bit 7 to a full-byte mask for the byte-wide vbsl selects.
    let hitm_a = O::sshr8(b1base_a);
    let hitm_b = O::sshr8(b1base_b);
    let one = O::dup_byte(0x01);
    let t3f = O::dup_byte(0x3F);
    let eighty = O::dup_byte(0x80);
    let clr15 = O::const16(CLR15);
    let clr0 = O::const16(CLR0);
    let next_a = O::vext(g_a, g_a, 1);
    let next_b = O::vext(g_b, g_b, 1);
    let nb_a = O::band(O::shr8_n(next_a, 6), one);
    let nb_b = O::band(O::shr8_n(next_b, 6), one);
    let b1_a = O::bor(b1base_a, nb_a);
    let b1_b = O::bor(b1base_b, nb_b);
    let blend1_a = O::bsl(O::band(hitm_a, clr15), b1_a, g_a);
    let blend1_b = O::bsl(O::band(hitm_b, clr15), b1_b, g_b);
    let cf_a = O::band(O::vext(hitm_a, hitm_a, 15), clr0);
    let cf_b = O::band(O::vext(hitm_b, hitm_b, 15), clr0);
    let b2_a = O::bor(O::band(g_a, t3f), eighty);
    let b2_b = O::bor(O::band(g_b, t3f), eighty);
    let out_a = O::bsl(cf_a, b2_a, blend1_a);
    let out_b = O::bsl(cf_b, b2_b, blend1_b);

    O::store16(out_ptr.add(*out_pos), out_a);
    O::store16(out_ptr.add(*out_pos + 16), out_b);
    *out_pos += 32;
    *i += 28;
    true
}

/// Decode one 16-byte window (2 blocks) into two store-ready 64-bit lanes
/// (7 useful bytes + 1 garbage byte each; the garbage is overwritten by the
/// next block's store or lies beyond the final length) and the anomaly flag.
///
/// # Safety
/// `ptr` must point to 16 readable bytes at `ptr.add(i)`.
pub(crate) unsafe fn decode_window<O: NeonOps>(ptr: *const u8, i: usize) -> (u64, u64, bool) {
    let b = O::load16(ptr.add(i));

    // Masks: lead/continuation candidates.
    let bc0 = O::band(b, O::dup_byte(0xC0));
    let lead = O::ceq(bc0, O::dup_byte(0xC0));
    let cont = O::ceq(bc0, O::dup_byte(0x80));

    // Transform: groups pass through, leads become their illegal byte,
    // continuations become (lead & 1) << 6 | (b & 0x3F). Byte 0's missing
    // (lead & 1) bit is patched into the packed output scalar-wise.
    let next_src = O::band(O::vext(b, b, 15), O::const16(CLR0));
    let next = O::bor(
        O::band(O::shl8_n(next_src, 6), O::dup_byte(0x40)),
        O::band(b, O::dup_byte(0x3F)),
    );
    let illegal = O::tbl(O::const16(ILLEGAL_T), O::band(O::shr8_n(b, 2), O::dup_byte(0x0F)));
    let g = O::bsl(lead, illegal, b);
    let g = O::bsl(cont, next, g);

    // Pack: per 64-bit lane, byte-reverse the 8 groups, form 14-bit pairs in
    // 16-bit lanes, then combine pairs from 16-bit to 14-bit spacing:
    // S = g7 | g6<<7 | ... | g0<<49 (group 0 at the top). The output bytes
    // are S's 7 bytes in big-endian order, i.e. rev64(S << 8).
    let r = O::rev64(g);
    let pairs = O::bor(O::band(r, O::const16(P7F_LO)), O::band(O::shr16_n(r, 1), O::const16(P3F80)));
    let s = O::bor(
        O::band(pairs, O::const16(M3FFF)),
        O::bor(
            O::band(O::shr64_n(pairs, 2), O::const16(M2)),
            O::bor(
                O::band(O::shr64_n(pairs, 4), O::const16(M4)),
                O::band(O::shr64_n(pairs, 6), O::const16(M6)),
            ),
        ),
    );
    let ready = O::rev64(O::shlq_u64(s, 8));

    // Straddle patch: byte 0's continuation lead is the previous window's
    // last byte (bit 6 of group 0 = bit 55 of the packed stream = bit 7 of
    // the first lane's byte 0 in the store-ready form).
    let prev = if i > 0 { unsafe { *ptr.add(i - 1) } } else { 0 };
    let b0 = unsafe { *ptr.add(i) };
    let patch0 = ((b0 & 0xC0) == 0x80 && (prev & 1) != 0) as u64;
    let l0 = O::get_lane_u64(ready, 0) | (patch0 << 7);
    let l1 = O::get_lane_u64(ready, 1);

    // Structure errors: a lead must be valid, every lead must be followed by
    // a continuation, every continuation must follow a lead. Byte 0's
    // continuation partner is the previous window's last byte and byte 15's
    // partner is byte 16 (outside the window); both checked in scalar code.
    let valid = O::ceq(O::band(b, O::dup_byte(0xE2)), O::dup_byte(0xC2));
    let short = O::ceq(O::band(b, O::dup_byte(0xFE)), O::dup_byte(0xDE));
    let err_inv = O::bic(lead, valid);
    let err_cont = O::band(O::bic(cont, O::vext(lead, lead, 15)), O::const16(CLR0));
    let err_pair = O::band(O::bic(lead, O::vext(cont, cont, 1)), O::const16(CLR15));
    let err = O::bor(O::bor(err_inv, err_cont), O::bor(err_pair, short));
    let bad = O::maxv(err) != 0 || (b0 & 0xC0) == 0x80 && (prev & 0xE2) != 0xC2;

    (l0, l1, bad)
}

/// Decode 16-byte windows until an anomaly requires the scalar codec.
/// Returns the number of input bytes consumed (a multiple of 16), whether the
/// last consumed byte was an escape lead, and the anomaly flag.
///
/// # Safety
/// `ptr` must point to `len` readable bytes; `out_ptr` must have capacity for
/// `len * 7 / 8 + 8` bytes.
pub(crate) unsafe fn decode_blocks<O: NeonOps>(
    ptr: *const u8,
    len: usize,
    out_ptr: *mut u8,
) -> (usize, bool, bool) {
    let mut i = 0usize;
    let mut out_pos = 0usize;
    let mut bad = false;

    // Keep the last 32 bytes for the scalar tail: the stream's final escape
    // pair may be a shortened marker (0xDE/0xDF), which the encoder only
    // emits at the very end and which the vector path cannot process.
    while i + 32 <= len {
        let (l0, l1, wbad) = decode_window::<O>(ptr, i);
        unsafe {
            (out_ptr.add(out_pos) as *mut u64).write_unaligned(l0);
            (out_ptr.add(out_pos + 7) as *mut u64).write_unaligned(l1);
        }
        out_pos += 14;
        bad |= wbad;
        i += 16;
    }

    let last_lead = i > 0 && unsafe { *ptr.add(i - 1) } & 0xE2 == 0xC2;
    (i, last_lead, bad)
}




#[cfg(test)]

#[cfg(test)]
pub(crate) mod sim;
#[cfg(target_arch = "aarch64")]
pub(crate) mod intr;
#[cfg(test)]
mod tests;
