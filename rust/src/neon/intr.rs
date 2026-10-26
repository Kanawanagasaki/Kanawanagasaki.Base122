    use super::*;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct Intr;

    impl NeonOps for Intr {
        type Reg = uint8x16_t;

        #[inline(always)]
        fn load64_dup(ptr: *const u8) -> uint8x16_t {
            unsafe { vreinterpretq_u8_u64(vld1q_dup_u64(ptr as *const u64)) }
        }
        #[inline(always)]
        fn load16(ptr: *const u8) -> uint8x16_t {
            unsafe { vld1q_u8(ptr) }
        }
        #[inline(always)]
        fn store16(ptr: *mut u8, r: uint8x16_t) {
            unsafe { vst1q_u8(ptr, r) }
        }
        #[inline(always)]
        fn shlq_u64c(r: uint8x16_t, counts: [i64; 2]) -> uint8x16_t {
            unsafe {
                vreinterpretq_u8_u64(vshlq_u64(
                    vreinterpretq_u64_u8(r),
                    vld1q_s64(counts.as_ptr()),
                ))
            }
        }
        #[inline(always)]
        fn tbl(table: uint8x16_t, idx: uint8x16_t) -> uint8x16_t {
            unsafe { vqtbl1q_u8(table, idx) }
        }
        #[inline(always)]
        fn tbl4(t0: uint8x16_t, t1: uint8x16_t, t2: uint8x16_t, t3: uint8x16_t, idx: uint8x16_t) -> uint8x16_t {
            let table = core::arch::aarch64::uint8x16x4_t(t0, t1, t2, t3);
            unsafe { vqtbl4q_u8(table, idx) }
        }
        #[inline(always)]
        fn vext(a: uint8x16_t, b: uint8x16_t, n: usize) -> uint8x16_t {
            unsafe {
                match n {
                    1 => vextq_u8::<1>(a, b),
                    8 => vextq_u8::<8>(a, b),
                    15 => vextq_u8::<15>(a, b),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn bsl(mask: uint8x16_t, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
            // Bitwise select; callers pass sshr8-extended full-byte masks.
            unsafe { vbslq_u8(mask, a, b) }
        }
        #[inline(always)]
        fn ceq(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
            unsafe { vceqq_u8(a, b) }
        }
        #[inline(always)]
        fn band(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
            unsafe { vandq_u8(a, b) }
        }
        #[inline(always)]
        fn bor(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
            unsafe { vorrq_u8(a, b) }
        }
        #[inline(always)]
        fn bic(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
            unsafe { vbicq_u8(a, b) }
        }
        #[inline(always)]
        fn shr8_n(r: uint8x16_t, n: u8) -> uint8x16_t {
            unsafe {
                match n {
                    1 => vshrq_n_u8::<1>(r),
                    2 => vshrq_n_u8::<2>(r),
                    4 => vshrq_n_u8::<4>(r),
                    6 => vshrq_n_u8::<6>(r),
                    7 => vshrq_n_u8::<7>(r),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn sshr8(r: uint8x16_t) -> uint8x16_t {
            // vtstq_u8: test bit 7 against 0x80, producing full 0xFF/0x00
            // masks (same result as an arithmetic shift right by 7).
            unsafe { vtstq_u8(r, vdupq_n_u8(0x80)) }
        }
        #[inline(always)]
        fn shl8_n(r: uint8x16_t, n: u8) -> uint8x16_t {
            unsafe {
                match n {
                    6 => vshlq_n_u8::<6>(r),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn shr16_n(r: uint8x16_t, n: u8) -> uint8x16_t {
            unsafe {
                match n {
                    1 => vreinterpretq_u8_u16(vshrq_n_u16::<1>(vreinterpretq_u16_u8(r))),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn shr64_n(r: uint8x16_t, n: u8) -> uint8x16_t {
            unsafe {
                match n {
                    2 => vreinterpretq_u8_u64(vshrq_n_u64::<2>(vreinterpretq_u64_u8(r))),
                    4 => vreinterpretq_u8_u64(vshrq_n_u64::<4>(vreinterpretq_u64_u8(r))),
                    6 => vreinterpretq_u8_u64(vshrq_n_u64::<6>(vreinterpretq_u64_u8(r))),
                    8 => vreinterpretq_u8_u64(vshrq_n_u64::<8>(vreinterpretq_u64_u8(r))),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn shlq_u64(r: uint8x16_t, n: u8) -> uint8x16_t {
            unsafe {
                match n {
                    1 => vreinterpretq_u8_u64(vshlq_n_u64::<1>(vreinterpretq_u64_u8(r))),
                    8 => vreinterpretq_u8_u64(vshlq_n_u64::<8>(vreinterpretq_u64_u8(r))),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn rev64(r: uint8x16_t) -> uint8x16_t {
            unsafe { vrev64q_u8(r) }
        }
        #[inline(always)]
        fn dup_byte(v: u8) -> uint8x16_t {
            unsafe { vdupq_n_u8(v) }
        }
        #[inline(always)]
        fn const16(b: [u8; 16]) -> uint8x16_t {
            unsafe { vld1q_u8(b.as_ptr()) }
        }
        #[inline(always)]
        fn get_lane_u8(r: uint8x16_t, j: usize) -> u8 {
            unsafe {
                match j {
                    15 => vgetq_lane_u8::<15>(r),
                    _ => unreachable!(),
                }
            }
        }
        #[inline(always)]
        fn maxv(r: uint8x16_t) -> u8 {
            unsafe { vmaxvq_u8(r) }
        }
        #[inline(always)]
        fn get_lane_u64(r: uint8x16_t, j: usize) -> u64 {
            unsafe {
                match j {
                    0 => vgetq_lane_u64::<0>(vreinterpretq_u64_u8(r)),
                    1 => vgetq_lane_u64::<1>(vreinterpretq_u64_u8(r)),
                    _ => unreachable!(),
                }
            }
        }
    }
