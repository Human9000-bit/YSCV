//! Contiguous RGB8 -> BGRA8 rows. Resampling uses the cached gather map instead.
#![allow(unsafe_code)]

pub(super) fn select_bgra() -> fn(&[u8], &mut [u8]) {
    #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
    if yscv_cpu::host_cpu().features.ssse3 {
        return |src, dst| {
            // SAFETY: selected only with SSSE3; the kernel bounds every load/store.
            unsafe { bgra_ssse3(src, dst) }
        };
    }
    #[cfg(all(
        not(miri),
        any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7"))
    ))]
    if yscv_cpu::host_cpu().features.neon {
        return |src, dst| {
            // SAFETY: selected only with NEON; the kernel bounds every load/store.
            unsafe { bgra_neon(src, dst) }
        };
    }
    bgra_scalar
}

fn bgra_scalar(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0)
    {
        *d = [s[2], s[1], s[0], 255];
    }
}

#[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "ssse3")]
unsafe fn bgra_ssse3(src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let shuffle = _mm_setr_epi8(2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128);
    let alpha = _mm_set1_epi32(0xff000000u32 as i32);
    let mut x = 0;
    while x + 16 <= dst.len() && x / 4 * 3 + 16 <= src.len() {
        // SAFETY: both unaligned 16-byte accesses are within the checked slices.
        unsafe {
            let rgb = _mm_loadu_si128(src.as_ptr().add(x / 4 * 3).cast());
            let bgra = _mm_or_si128(_mm_shuffle_epi8(rgb, shuffle), alpha);
            _mm_storeu_si128(dst.as_mut_ptr().add(x).cast(), bgra);
        }
        x += 16;
    }
    bgra_scalar(&src[x / 4 * 3..], &mut dst[x..]);
}

#[cfg(all(
    not(miri),
    any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7"))
))]
#[target_feature(enable = "neon")]
unsafe fn bgra_neon(src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "arm")]
    use std::arch::arm::*;
    let mut x = 0;
    while x + 64 <= dst.len() && x / 4 * 3 + 48 <= src.len() {
        // SAFETY: vld3 reads 48 bytes and vst4 writes 64, within the checked slices.
        unsafe {
            let rgb = vld3q_u8(src.as_ptr().add(x / 4 * 3));
            vst4q_u8(
                dst.as_mut_ptr().add(x),
                uint8x16x4_t(rgb.2, rgb.1, rgb.0, vdupq_n_u8(255)),
            );
        }
        x += 64;
    }
    bgra_scalar(&src[x / 4 * 3..], &mut dst[x..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatched_rows_match_scalar_at_every_tail_and_alignment() {
        let convert = select_bgra();
        for width in 0..=129 {
            for offset in 0..16 {
                let src: Vec<u8> = (0..width * 3 + offset).map(|i| (i * 37) as u8).collect();
                let mut got = vec![19; width * 4 + offset + 7];
                let mut expected = got.clone();
                convert(&src[offset..], &mut got[offset..offset + width * 4]);
                bgra_scalar(&src[offset..], &mut expected[offset..offset + width * 4]);
                assert_eq!(got, expected, "width {width}, offset {offset}");
            }
        }
    }
}
