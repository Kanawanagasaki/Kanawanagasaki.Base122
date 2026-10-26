    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct Reg(pub [u8; 16]);

    impl Reg {
        pub fn lane0_u64(self) -> u64 {
            u64::from_le_bytes(self.0[0..8].try_into().unwrap())
        }
        pub fn lane1_u64(self) -> u64 {
            u64::from_le_bytes(self.0[8..16].try_into().unwrap())
        }
    }

    #[derive(Clone, Copy)]
    pub struct Sim;

    fn zip(a: Reg, b: Reg, f: fn(u8, u8) -> u8) -> [u8; 16] {
        let mut out = [0u8; 16];
        for (j, o) in out.iter_mut().enumerate() {
            *o = f(a.0[j], b.0[j]);
        }
        out
    }

    impl NeonOps for Sim {
        type Reg = Reg;

        #[inline(always)]
        fn load64_dup(ptr: *const u8) -> Reg {
            let mut b = [0u8; 16];
            unsafe { core::ptr::copy_nonoverlapping(ptr, b.as_mut_ptr(), 8) };
            let v = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let vb = v.to_le_bytes();
            let mut out = [0u8; 16];
            out[0..8].copy_from_slice(&vb);
            out[8..16].copy_from_slice(&vb);
            Reg(out)
        }

        #[inline(always)]
        fn load16(ptr: *const u8) -> Reg {
            let mut b = [0u8; 16];
            unsafe { core::ptr::copy_nonoverlapping(ptr, b.as_mut_ptr(), 16) };
            Reg(b)
        }

        #[inline(always)]
        fn store16(ptr: *mut u8, r: Reg) {
            unsafe { core::ptr::copy_nonoverlapping(r.0.as_ptr(), ptr, 16) };
        }


        #[inline(always)]
        fn shlq_u64c(r: Reg, counts: [i64; 2]) -> Reg {
            let mut out = [0u8; 16];
            for lane in 0..2 {
                let v = u64::from_le_bytes(r.0[8 * lane..8 * lane + 8].try_into().unwrap());
                let c = counts[lane];
                let s = if c >= 64 || c <= -64 {
                    0
                } else if c >= 0 {
                    v << c
                } else {
                    v >> (-c)
                };
                out[8 * lane..8 * lane + 8].copy_from_slice(&s.to_le_bytes());
            }
            Reg(out)
        }

        #[inline(always)]
        fn tbl(table: Reg, idx: Reg) -> Reg {
            // vqtbl1q_u8: index bit 7 set or index >= 16 yields 0.
            let mut out = [0u8; 16];
            for (j, &ix) in idx.0.iter().enumerate() {
                out[j] = if ix & 0x80 != 0 || ix >= 16 { 0 } else { table.0[ix as usize] };
            }
            Reg(out)
        }

        #[inline(always)]
        fn tbl4(t0: Reg, t1: Reg, t2: Reg, t3: Reg, idx: Reg) -> Reg {
            let mut virt = [0u8; 64];
            virt[0..16].copy_from_slice(&t0.0);
            virt[16..32].copy_from_slice(&t1.0);
            virt[32..48].copy_from_slice(&t2.0);
            virt[48..64].copy_from_slice(&t3.0);
            let mut out = [0u8; 16];
            for (j, &ix) in idx.0.iter().enumerate() {
                out[j] = if ix & 0x80 != 0 || ix >= 64 { 0 } else { virt[ix as usize] };
            }
            Reg(out)
        }

        #[inline(always)]
        fn vext(a: Reg, b: Reg, n: usize) -> Reg {
            // vextq_u8(a, b, n) = {a[n..16], b[0..n]}.
            let mut out = [0u8; 16];
            for (j, o) in out.iter_mut().enumerate() {
                *o = if j < 16 - n { a.0[j + n] } else { b.0[j - (16 - n)] };
            }
            Reg(out)
        }

        #[inline(always)]
        fn bsl(mask: Reg, a: Reg, b: Reg) -> Reg {
            // vbslq_u8 contract: bitwise select; callers must pass full-byte
            // 0xFF/0x00 masks (sshr8-extended), which makes it a byte select.
            let mut out = [0u8; 16];
            for (j, o) in out.iter_mut().enumerate() {
                *o = if mask.0[j] != 0 { a.0[j] } else { b.0[j] };
            }
            Reg(out)
        }

        #[inline(always)]
        fn ceq(a: Reg, b: Reg) -> Reg {
            let mut out = [0u8; 16];
            for (j, o) in out.iter_mut().enumerate() {
                *o = if a.0[j] == b.0[j] { 0xFF } else { 0x00 };
            }
            Reg(out)
        }

        #[inline(always)]
        fn band(a: Reg, b: Reg) -> Reg {
            Reg(zip(a, b, |x, y| x & y))
        }
        #[inline(always)]
        fn bor(a: Reg, b: Reg) -> Reg {
            Reg(zip(a, b, |x, y| x | y))
        }
        #[inline(always)]
        fn bic(a: Reg, b: Reg) -> Reg {
            Reg(zip(a, b, |x, y| x & !y))
        }

        #[inline(always)]
        fn shr8_n(r: Reg, n: u8) -> Reg {
            Reg(r.0.map(|x| x >> n))
        }
        #[inline(always)]
        fn shl8_n(r: Reg, n: u8) -> Reg {
            Reg(r.0.map(|x| x << n))
        }

        #[inline(always)]
        fn sshr8(r: Reg) -> Reg {
            Reg(r.0.map(|x| if x & 0x80 != 0 { 0xFF } else { 0x00 }))
        }

        #[inline(always)]
        fn shr16_n(r: Reg, n: u8) -> Reg {
            let mut out = [0u8; 16];
            for lane in 0..8 {
                let v = u16::from_le_bytes(r.0[2 * lane..2 * lane + 2].try_into().unwrap());
                out[2 * lane..2 * lane + 2].copy_from_slice(&(v >> n).to_le_bytes());
            }
            Reg(out)
        }

        #[inline(always)]
        fn shr64_n(r: Reg, n: u8) -> Reg {
            let a = r.lane0_u64() >> n;
            let b = r.lane1_u64() >> n;
            let mut out = [0u8; 16];
            out[0..8].copy_from_slice(&a.to_le_bytes());
            out[8..16].copy_from_slice(&b.to_le_bytes());
            Reg(out)
        }

        #[inline(always)]
        fn shlq_u64(r: Reg, n: u8) -> Reg {
            let a = r.lane0_u64() << n;
            let b = r.lane1_u64() << n;
            let mut out = [0u8; 16];
            out[0..8].copy_from_slice(&a.to_le_bytes());
            out[8..16].copy_from_slice(&b.to_le_bytes());
            Reg(out)
        }

        #[inline(always)]
        fn rev64(r: Reg) -> Reg {
            let mut out = [0u8; 16];
            for lane in 0..2 {
                for j in 0..8 {
                    out[8 * lane + j] = r.0[8 * lane + 7 - j];
                }
            }
            Reg(out)
        }

        #[inline(always)]
        fn dup_byte(v: u8) -> Reg {
            Reg([v; 16])
        }

        #[inline(always)]
        fn const16(b: [u8; 16]) -> Reg {
            Reg(b)
        }

        #[inline(always)]
        fn get_lane_u8(r: Reg, j: usize) -> u8 {
            r.0[j]
        }

        #[inline(always)]
        fn maxv(r: Reg) -> u8 {
            r.0.iter().copied().max().unwrap_or(0)
        }

        #[inline(always)]
        fn get_lane_u64(r: Reg, j: usize) -> u64 {
            if j == 0 { r.lane0_u64() } else { r.lane1_u64() }
        }
    }
