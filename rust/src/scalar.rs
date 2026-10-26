//! Scalar SWAR codec: the portable baseline used directly on targets without
//! a SIMD kernel and as the fallback/tail for the AVX2 and NEON paths.

// Bit-gather helper shared with the AVX2 kernel.
pub(crate) const LO7_8: u64 = 0x7F7F_7F7F_7F7F_7F7F;

pub(crate) const ILLEGALS: [u8; 6] = [0, 10, 13, 34, 38, 92];
const SHORTENED: u8 = 0b111;
const ASCII_MASK_8: u64 = 0x8080_8080_8080_8080;
const Z0_MASK: u64 = 0x0000_0000_0000_0000;
const Z10_MASK: u64 = 0x0A0A_0A0A_0A0A_0A0A;
const Z13_MASK: u64 = 0x0D0D_0D0D_0D0D_0D0D;
const Z34_MASK: u64 = 0x2222_2222_2222_2222;
const Z38_MASK: u64 = 0x2626_2626_2626_2626;
const Z92_MASK: u64 = 0x5C5C_5C5C_5C5C_5C5C;

pub(crate) const CLASS: [u8; 128] = {
    let mut arr = [0u8; 128];
    arr[0] = 1;
    arr[10] = 2;
    arr[13] = 3;
    arr[34] = 4;
    arr[38] = 5;
    arr[92] = 6;
    arr
};

const ESCAPE_TABLE: [[u16; 128]; 7] = {
    let mut table = [[0u16; 128]; 7];
    let mut idx = 0usize;
    while idx < 6 {
        let mut next = 0usize;
        while next < 128 {
            let n = next as u8;
            let b1 = 0b1100_0010 | (idx as u8) << 2 | (n >> 6);
            let b2 = 0x80 | (n & 0x3F);
            table[idx][next] = u16::from_le_bytes([b1, b2]);
            next += 1;
        }
        idx += 1;
    }
    let mut next = 0usize;
    while next < 128 {
        let n = next as u8;
        let b1 = 0b1100_0010 | (SHORTENED << 2) | (n >> 6);
        let b2 = 0x80 | (n & 0x3F);
        table[6][next] = u16::from_le_bytes([b1, b2]);
        next += 1;
    }
    table
};

/// Direct escape-pair lookup: `ESCAPE_PAIR_TABLE[((class - 1) << 7) | next]`
/// yields the little-endian 2-byte escape for an illegal group's class (1-6)
/// and the following 7-bit group.
const ESCAPE_PAIR_TABLE: [u16; 6 * 128] = {
    let mut table = [0u16; 6 * 128];
    let mut idx = 0usize;
    while idx < 6 {
        let mut next = 0usize;
        while next < 128 {
            let n = next as u8;
            let b1 = 0b1100_0010 | ((idx as u8) << 2) | (n >> 6);
            let b2 = 0x80 | (n & 0x3F);
            table[(idx << 7) | next] = u16::from_le_bytes([b1, b2]);
            next += 1;
        }
        idx += 1;
    }
    table
};

pub(crate) const LEAD_DECODE: [u8; 256] = {
    let mut arr = [0xFFu8; 256];

    let mut idx = 0usize;
    while idx < 6 {
        let mut bit = 0usize;
        while bit < 2 {
            let lead = 0b1100_0010 | ((idx as u8) << 2) | (bit as u8);
            arr[lead as usize] = ((idx as u8) << 1) | (bit as u8);
            bit += 1;
        }
        idx += 1;
    }

    let mut bit = 0usize;
    while bit < 2 {
        let lead = 0b1100_0010 | (SHORTENED << 2) | (bit as u8);
        arr[lead as usize] = (SHORTENED << 1) | (bit as u8);
        bit += 1;
    }

    arr
};

#[inline(always)]
unsafe fn store_u64_le(dst: *mut u8, value: u64) {
    unsafe { (dst as *mut u64).write_unaligned(value.to_le()) };
}

#[inline(always)]
unsafe fn store_u16_le(dst: *mut u8, value: u16) {
    unsafe { (dst as *mut u16).write_unaligned(value.to_le()) };
}

#[inline(always)]
unsafe fn store_be_partial(dst: *mut u8, value: u64, len: usize) {
    debug_assert!(len > 0 && len <= 8);
    let shifted = value << ((8 - len) * 8);
    unsafe { (dst as *mut u64).write_unaligned(shifted.to_be()) };
}

#[inline(always)]
unsafe fn load_u64_le(ptr: *const u8) -> u64 {
    u64::from_le(unsafe { (ptr as *const u64).read_unaligned() })
}

#[inline(always)]
pub(crate) unsafe fn load56_be_overread1(ptr: *const u8) -> u64 {
    u64::from_be(unsafe { (ptr as *const u64).read_unaligned() }) >> 8
}

#[inline(always)]
unsafe fn load56_be_exact(ptr: *const u8) -> u64 {
    #[cfg(miri)]
    {
        let mut tmp = 0u64;
        unsafe { core::ptr::copy_nonoverlapping(ptr, &mut tmp as *mut u64 as *mut u8, 7) };
        u64::from_be(tmp) >> 8
    }
    #[cfg(not(miri))]
    {
        unsafe {
            let val = (ptr as *const u64).read_unaligned();
            val.swap_bytes() >> 8
        }
    }
}

#[inline(always)]
pub(crate) fn split56_to_groups_le(bits56: u64) -> u64 {
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "bmi2"
    ))]
    {
        let scattered = unsafe { core::arch::x86_64::_pdep_u64(bits56, LO7_8) };
        return scattered.swap_bytes();
    }
    #[allow(unreachable_code)]
    {
        ((bits56 >> 49) & 0x7F)
            | (((bits56 >> 42) & 0x7F) << 8)
            | (((bits56 >> 35) & 0x7F) << 16)
            | (((bits56 >> 28) & 0x7F) << 24)
            | (((bits56 >> 21) & 0x7F) << 32)
            | (((bits56 >> 14) & 0x7F) << 40)
            | (((bits56 >> 7) & 0x7F) << 48)
            | ((bits56 & 0x7F) << 56)
    }
}

#[inline(always)]
fn gather_groups_to_bits56(chunk_le: u64) -> u64 {
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "bmi2"
    ))]
    {
        return unsafe { core::arch::x86_64::_pext_u64(chunk_le.swap_bytes(), LO7_8) };
    }
    #[allow(unreachable_code)]
    {
        let g0 = chunk_le & 0xFF;
        let g1 = (chunk_le >> 8) & 0xFF;
        let g2 = (chunk_le >> 16) & 0xFF;
        let g3 = (chunk_le >> 24) & 0xFF;
        let g4 = (chunk_le >> 32) & 0xFF;
        let g5 = (chunk_le >> 40) & 0xFF;
        let g6 = (chunk_le >> 48) & 0xFF;
        let g7 = (chunk_le >> 56) & 0xFF;

        (g0 << 49) | (g1 << 42) | (g2 << 35) | (g3 << 28) | (g4 << 21) | (g5 << 14) | (g6 << 7) | g7
    }
}

#[inline(always)]
fn illegal_high_mask(x: u64) -> u64 {
    let z0 = x ^ Z0_MASK;
    let z10 = x ^ Z10_MASK;
    let z13 = x ^ Z13_MASK;
    let z34 = x ^ Z34_MASK;
    let z38 = x ^ Z38_MASK;
    let z92 = x ^ Z92_MASK;

    let t0 = !z0.wrapping_add(LO7_8);
    let t10 = !z10.wrapping_add(LO7_8);
    let t13 = !z13.wrapping_add(LO7_8);
    let t34 = !z34.wrapping_add(LO7_8);
    let t38 = !z38.wrapping_add(LO7_8);
    let t92 = !z92.wrapping_add(LO7_8);

    (t0 | t10 | t13 | t34 | t38 | t92) & ASCII_MASK_8
}

#[inline(always)]
unsafe fn emit_escape_pair(out_ptr: *mut u8, out_pos: &mut usize, first: u8, next: u8) {
    let class = unsafe { *CLASS.get_unchecked(first as usize) };
    debug_assert!((1..=6).contains(&class));
    let idx = (class - 1) as usize;
    let pair = unsafe { *ESCAPE_TABLE.get_unchecked(idx).get_unchecked(next as usize) };
    unsafe { store_u16_le(out_ptr.add(*out_pos), pair) };
    *out_pos += 2;
}

#[inline(always)]
unsafe fn emit_shortened(out_ptr: *mut u8, out_pos: &mut usize, bits: u8) {
    let pair = unsafe { *ESCAPE_TABLE.get_unchecked(6).get_unchecked(bits as usize) };
    unsafe { store_u16_le(out_ptr.add(*out_pos), pair) };
    *out_pos += 2;
}

#[inline(always)]
fn pull7_tail(tail: &[u8], pos: &mut usize, acc: &mut u64, acc_bits: &mut u32) -> Option<u8> {
    while *acc_bits < 7 && *pos < tail.len() {
        *acc = (*acc << 8) | tail[*pos] as u64;
        *pos += 1;
        *acc_bits += 8;
    }

    if *acc_bits >= 7 {
        *acc_bits -= 7;
        let bits = ((*acc >> *acc_bits) & 0x7F) as u8;
        if *acc_bits == 0 {
            *acc = 0;
        } else {
            *acc &= (1_u64 << *acc_bits) - 1;
        }
        Some(bits)
    } else if *acc_bits > 0 {
        let bits = ((*acc << (7 - *acc_bits)) & 0x7F) as u8;
        *acc = 0;
        *acc_bits = 0;
        Some(bits)
    } else {
        None
    }
}

#[inline(always)]
fn group_count(input_len: usize) -> usize {
    if input_len == 0 {
        0
    } else {
        input_len.saturating_mul(8).saturating_add(6) / 7
    }
}

#[inline(always)]
pub(crate) fn encoded_capacity(input_len: usize) -> usize {
    group_count(input_len).saturating_mul(2).saturating_add(8)
}

#[inline(always)]
pub(crate) fn decoded_capacity(encoded_len: usize) -> usize {
    encoded_len
        .saturating_mul(7)
        .saturating_add(7)
        .saturating_div(8)
        .saturating_add(8)
}

#[inline(always)]
unsafe fn process_groups8_masked(
    mut groups_le: u64,
    mut illegal_mask: u64,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    pending_illegal_bits: &mut u8,
    has_pending_illegal: &mut bool,
) {
    let mut remaining = 8usize;

    if *has_pending_illegal {
        let next = groups_le as u8;
        unsafe { emit_escape_pair(out_ptr, out_pos, *pending_illegal_bits, next) };
        *has_pending_illegal = false;
        groups_le >>= 8;
        illegal_mask >>= 8;
        remaining -= 1;
    }

    if illegal_mask == 0 {
        unsafe { store_u64_le(out_ptr.add(*out_pos), groups_le) };
        *out_pos += remaining;
        return;
    }

    // Branchless splice loop: every escape that has its next group inside this
    // chunk (p <= remaining - 2) is spliced into the output value. An escape
    // consumes two groups and emits two bytes, so the chunk output stays
    // `remaining` bytes; consecutive escapes compose independently.
    let mut v = groups_le;
    loop {
        let p = (illegal_mask.trailing_zeros() >> 3) as usize;
        if p > remaining - 2 {
            break;
        }
        let cur = (v >> (p * 8)) as u8;
        let next = (v >> ((p + 1) * 8)) as u8;
        let class = unsafe { *CLASS.get_unchecked(cur as usize) };
        let pair = unsafe {
            *ESCAPE_PAIR_TABLE.get_unchecked((((class - 1) as usize) << 7) | (next as usize))
        } as u64;
        let low_mask = (1u64 << (p * 8)).wrapping_sub(1);
        let high_mask = if p + 2 < remaining {
            !0u64 << ((p + 2) * 8)
        } else {
            0
        };
        v = (v & low_mask) | (pair << (p * 8)) | (v & high_mask);
        illegal_mask &= !((1u64 << (p * 8 + 7)) | (1u64 << ((p + 1) * 8 + 7)));
    }

    if illegal_mask != 0 {
        // The chunk's last group is illegal; its escape needs the next group
        // from the following chunk, so it is deferred as a pending escape.
        let p = remaining - 1;
        let cur = (v >> (p * 8)) as u8;
        *pending_illegal_bits = cur;
        *has_pending_illegal = true;
        // Store the value as-is; the pending pair emitted at the start of the
        // next chunk overwrites the stored last byte.
        unsafe { store_u64_le(out_ptr.add(*out_pos), v) };
        *out_pos += remaining - 1;
        return;
    }

    unsafe { store_u64_le(out_ptr.add(*out_pos), v) };
    *out_pos += remaining;
}

/// Scalar SWAR processing of one 28-byte block (4 chunks of 8 groups).
#[inline(always)]
pub(crate) unsafe fn scalar_block_28(
    ptr: *const u8,
    _len: usize,
    i: &mut usize,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    pending_illegal_bits: &mut u8,
    has_pending_illegal: &mut bool,
) {
    let bits56_a = unsafe { load56_be_overread1(ptr.add(*i)) };
    let bits56_b = unsafe { load56_be_overread1(ptr.add(*i + 7)) };
    let bits56_c = unsafe { load56_be_overread1(ptr.add(*i + 14)) };
    let bits56_d = unsafe { load56_be_overread1(ptr.add(*i + 21)) };

    let groups_a = split56_to_groups_le(bits56_a);
    let groups_b = split56_to_groups_le(bits56_b);
    let groups_c = split56_to_groups_le(bits56_c);
    let groups_d = split56_to_groups_le(bits56_d);

    let mask_a = illegal_high_mask(groups_a);
    let mask_b = illegal_high_mask(groups_b);
    let mask_c = illegal_high_mask(groups_c);
    let mask_d = illegal_high_mask(groups_d);

    if !*has_pending_illegal && (mask_a | mask_b | mask_c | mask_d) == 0 {
        unsafe {
            store_u64_le(out_ptr.add(*out_pos), groups_a);
            store_u64_le(out_ptr.add(*out_pos + 8), groups_b);
            store_u64_le(out_ptr.add(*out_pos + 16), groups_c);
            store_u64_le(out_ptr.add(*out_pos + 24), groups_d);
        }
        *out_pos += 32;
        *i += 28;
        return;
    }

    unsafe {
        process_groups8_masked(
            groups_a, mask_a, out_ptr, out_pos, pending_illegal_bits, has_pending_illegal,
        );
        process_groups8_masked(
            groups_b, mask_b, out_ptr, out_pos, pending_illegal_bits, has_pending_illegal,
        );
        process_groups8_masked(
            groups_c, mask_c, out_ptr, out_pos, pending_illegal_bits, has_pending_illegal,
        );
        process_groups8_masked(
            groups_d, mask_d, out_ptr, out_pos, pending_illegal_bits, has_pending_illegal,
        );
    }
    *i += 28;
}

/// Scalar SWAR processing of one 14-byte block (2 chunks of 8 groups).
#[inline(always)]
pub(crate) unsafe fn scalar_block_14(
    ptr: *const u8,
    _len: usize,
    i: &mut usize,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    pending_illegal_bits: &mut u8,
    has_pending_illegal: &mut bool,
) {
    let bits56_a = unsafe { load56_be_overread1(ptr.add(*i)) };
    let bits56_b = unsafe { load56_be_overread1(ptr.add(*i + 7)) };

    let groups_a = split56_to_groups_le(bits56_a);
    let groups_b = split56_to_groups_le(bits56_b);

    let mask_a = illegal_high_mask(groups_a);
    let mask_b = illegal_high_mask(groups_b);

    if !*has_pending_illegal && (mask_a | mask_b) == 0 {
        unsafe {
            store_u64_le(out_ptr.add(*out_pos), groups_a);
            store_u64_le(out_ptr.add(*out_pos + 8), groups_b);
        }
        *out_pos += 16;
        *i += 14;
        return;
    }

    unsafe {
        process_groups8_masked(
            groups_a,
            mask_a,
            out_ptr,
            out_pos,
            pending_illegal_bits,
            has_pending_illegal,
        );
        process_groups8_masked(
            groups_b,
            mask_b,
            out_ptr,
            out_pos,
            pending_illegal_bits,
            has_pending_illegal,
        );
    }
    *i += 14;
}

/// Scalar tail of `encode_into`: 14/7-byte blocks and the final bit stream.
/// Shared by the vector paths so their simulation tests can drive the same
/// pipeline.
#[cold]
pub(crate) unsafe fn encode_tail(
    data: &[u8],
    mut i: usize,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    mut pending_illegal_bits: u8,
    mut has_pending_illegal: bool,
) -> usize {
    let len = data.len();
    let ptr = data.as_ptr();

    while i + 15 <= len {
        unsafe {
            scalar_block_14(
                ptr,
                len,
                &mut i,
                out_ptr,
                out_pos,
                &mut pending_illegal_bits,
                &mut has_pending_illegal,
            )
        };
    }

    while i + 8 <= len {
        let bits56 = unsafe { load56_be_overread1(ptr.add(i)) };
        let groups_le = split56_to_groups_le(bits56);
        let mask = illegal_high_mask(groups_le);

        if !has_pending_illegal && mask == 0 {
            unsafe { store_u64_le(out_ptr.add(*out_pos), groups_le) };
            *out_pos += 8;
            i += 7;
            continue;
        }

        unsafe {
            process_groups8_masked(
                groups_le,
                mask,
                out_ptr,
                out_pos,
                &mut pending_illegal_bits,
                &mut has_pending_illegal,
            );
        }
        i += 7;
    }

    if i + 7 <= len {
        let bits56 = unsafe { load56_be_exact(ptr.add(i)) };
        let groups_le = split56_to_groups_le(bits56);
        let mask = illegal_high_mask(groups_le);

        if !has_pending_illegal && mask == 0 {
            unsafe { store_u64_le(out_ptr.add(*out_pos), groups_le) };
            *out_pos += 8;
            i += 7;
        } else {
            unsafe {
                process_groups8_masked(
                    groups_le,
                    mask,
                    out_ptr,
                    out_pos,
                    &mut pending_illegal_bits,
                    &mut has_pending_illegal,
                );
            }
            i += 7;
        }
    }

    let tail = &data[i..];
    let mut tail_pos = 0usize;
    let mut acc = 0u64;
    let mut acc_bits = 0u32;

    if has_pending_illegal {
        if let Some(nb) = pull7_tail(tail, &mut tail_pos, &mut acc, &mut acc_bits) {
            unsafe { emit_escape_pair(out_ptr, out_pos, pending_illegal_bits, nb) };
        } else {
            unsafe { emit_shortened(out_ptr, out_pos, pending_illegal_bits) };
        }
    }

    while let Some(cur) = pull7_tail(tail, &mut tail_pos, &mut acc, &mut acc_bits) {
        let class = unsafe { *CLASS.get_unchecked(cur as usize) };
        if class == 0 {
            unsafe { *out_ptr.add(*out_pos) = cur };
            *out_pos += 1;
        } else if let Some(nb) = pull7_tail(tail, &mut tail_pos, &mut acc, &mut acc_bits) {
            unsafe { emit_escape_pair(out_ptr, out_pos, cur, nb) };
        } else {
            unsafe { emit_shortened(out_ptr, out_pos, cur) };
            break;
        }
    }

    *out_pos
}

/// Branchless push of exactly `p` groups (0..=8) packed in the low 7 bits of
/// each of the chunk's first `p` bytes. One combined accumulator update, so the
/// accumulator dependency chain stays short regardless of escape density.
#[inline(always)]
unsafe fn push_p_groups_le(
    chunk_le: u64,
    p: usize,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    acc: &mut u64,
    acc_bits: &mut u32,
) {
    let add_bits = (p as u32) * 7;
    // Mask each byte to its low 7 bits first: escape bytes (>= 0x80) would
    // otherwise pollute the neighbouring group slots in the gather. The gather
    // packs g0 at the top, so shifting right exposes the first `p` groups.
    let packed = gather_groups_to_bits56(chunk_le & LO7_8) >> (56 - add_bits as usize);
    let combined = (*acc << add_bits) | packed;
    let total_bits = *acc_bits + add_bits;
    let emitted = (total_bits >> 3) as usize;
    let new_bits = total_bits & 7;

    if emitted != 0 {
        unsafe { store_be_partial(out_ptr.add(*out_pos), combined >> new_bits, emitted) };
        *out_pos += emitted;
    }

    *acc_bits = new_bits;
    *acc = combined & ((1u64 << new_bits).wrapping_sub(1));
}

#[inline(always)]
pub(crate) unsafe fn push7_scalar(
    out_ptr: *mut u8,
    out_pos: &mut usize,
    acc: &mut u64,
    acc_bits: &mut u32,
    bits: u8,
) {    let combined = (*acc << 7) | (bits as u64);
    let total_bits = *acc_bits + 7;
    let emitted = total_bits >> 3;
    let new_bits = total_bits & 7;
    // Store unconditionally; when `emitted == 0` the byte is overwritten by the
    // next push (or lies beyond the final length).
    unsafe { *out_ptr.add(*out_pos) = (combined >> new_bits) as u8 };
    *out_pos += emitted as usize;
    *acc_bits = new_bits;
    *acc = combined & ((1u64 << new_bits).wrapping_sub(1));
}

#[inline(always)]
unsafe fn push14_scalar(
    out_ptr: *mut u8,
    out_pos: &mut usize,
    acc: &mut u64,
    acc_bits: &mut u32,
    hi7: u8,
    lo7: u8,
) {
    let combined = (*acc << 14) | ((hi7 as u64) << 7) | (lo7 as u64);
    let total_bits = *acc_bits + 14;
    let new_bits = total_bits & 7;
    let emitted = total_bits >> 3;
    let out_bits = combined >> new_bits;
    // `emitted` is 1 or 2. Shift the value so the big-endian u16 store always
    // writes the correct first byte; the trailing byte is overwritten by the
    // next push or lies beyond the final length.
    let v = (out_bits as u16) << (8 * (2 - emitted));
    unsafe { (out_ptr.add(*out_pos) as *mut u16).write_unaligned(v.to_be()) };
    *out_pos += emitted as usize;
    *acc_bits = new_bits;
    *acc = combined & ((1u64 << new_bits).wrapping_sub(1));
}

#[cold]
#[inline(never)]
fn decode_err<T>(msg: &'static str) -> Result<T, &'static str> {
    Err(msg)
}

/// Validate and consume the escape pair straddling the end of the vector
/// region: the last vector byte is a lead and its continuation is the first
/// byte of the scalar tail.
#[cfg(any(all(target_arch = "x86_64", target_feature = "avx2"), target_arch = "aarch64"))]
#[inline(always)]
pub(crate) unsafe fn handle_last_lead(
    ptr: *const u8,
    i: &mut usize,
    len: usize,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    acc: &mut u64,
    acc_bits: &mut u32,
) -> Result<(), &'static str> {
    if *i < len {
        let b1 = unsafe { *ptr.add(*i - 1) };
        let b2 = unsafe { *ptr.add(*i) };
        if (b2 & 0xC0) != 0x80 {
            return decode_err("Invalid continuation byte");
        }
        unsafe {
            push7_scalar(out_ptr, out_pos, acc, acc_bits, ((b1 & 1) << 6) | (b2 & 0x3F));
        }
        *i += 1;
        Ok(())
    } else {
        decode_err("Unexpected end of input")
    }
}

/// Scalar tail of `decode_into`: the 8-byte window loop and the per-byte
/// loop. Shared by the vector paths so their simulation tests can drive the
/// same pipeline.
#[inline(always)]
pub(crate) unsafe fn decode_tail(
    bytes: &[u8],
    mut i: usize,
    out_ptr: *mut u8,
    out_pos: &mut usize,
    acc: &mut u64,
    acc_bits: &mut u32,
) -> Result<(), &'static str> {
    let len = bytes.len();
    let ptr = bytes.as_ptr();

    while i + 8 <= len {
        let chunk = unsafe { load_u64_le(ptr.add(i)) };
        let p = (chunk & ASCII_MASK_8).trailing_zeros() >> 3;
        let p = p as usize;

        // Push the `p` clean groups (p == 8 is the all-ASCII fast path).
        unsafe {
            push_p_groups_le(chunk, p, out_ptr, out_pos, acc, acc_bits);
        }
        i += p;

        if p == 8 {
            continue;
        }

        // Escape sequence starting at byte `p` of the window.
        let b1 = (chunk >> (p * 8)) as u8;
        let code = LEAD_DECODE[b1 as usize];
        if code == 0xFF {
            return decode_err("Invalid lead byte");
        }
        if i + 2 > len {
            return decode_err("Unexpected end of input");
        }
        let b2 = unsafe { *ptr.add(i + 1) };
        if (b2 & 0xC0) != 0x80 {
            return decode_err("Invalid continuation byte");
        }
        i += 2;

        let illegal_index = code >> 1;
        let first_bit = code & 1;
        let next = (first_bit << 6) | (b2 & 0x3F);

        unsafe {
            if illegal_index < 6 {
                push14_scalar(
                    out_ptr,
                    out_pos,
                    acc,
                    acc_bits,
                    ILLEGALS[illegal_index as usize],
                    next,
                );
            } else {
                debug_assert_eq!(illegal_index, SHORTENED);
                push7_scalar(out_ptr, out_pos, acc, acc_bits, next);
            }
        }
    }

    while i < len {
        let b = unsafe { *ptr.add(i) };
        if b < 128 {
            i += 1;
            unsafe { push7_scalar(out_ptr, out_pos, acc, acc_bits, b) };
            continue;
        }

        let code = LEAD_DECODE[b as usize];
        if code == 0xFF {
            return decode_err("Invalid lead byte");
        }

        if i + 1 >= len {
            return decode_err("Unexpected end of input");
        }

        let b2 = unsafe { *ptr.add(i + 1) };
        if (b2 & 0xC0) != 0x80 {
            return decode_err("Invalid continuation byte");
        }

        i += 2;

        let illegal_index = code >> 1;
        let first_bit = code & 1;
        let next = (first_bit << 6) | (b2 & 0x3F);

        unsafe {
            if illegal_index < 6 {
                push14_scalar(
                    out_ptr,
                    out_pos,
                    acc,
                    acc_bits,
                    ILLEGALS[illegal_index as usize],
                    next,
                );
            } else {
                debug_assert_eq!(illegal_index, SHORTENED);
                push7_scalar(out_ptr, out_pos, acc, acc_bits, next);
            }
        }
    }

    Ok(())
}

