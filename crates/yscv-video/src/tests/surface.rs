use super::*;

const FORMATS: [SurfaceFormat; 3] = [
    SurfaceFormat::Rgb8,
    SurfaceFormat::Bgra8,
    SurfaceFormat::Rgb565Le,
];

fn reference_pixel(rgb: &[u8], format: SurfaceFormat) -> [u8; 4] {
    match format {
        SurfaceFormat::Rgb8 => [rgb[0], rgb[1], rgb[2], 0],
        SurfaceFormat::Bgra8 => [rgb[2], rgb[1], rgb[0], 255],
        SurfaceFormat::Rgb565Le => {
            let value = (u16::from(rgb[0]) / 8) * 2048
                + (u16::from(rgb[1]) / 4) * 32
                + u16::from(rgb[2]) / 8;
            let [low, high] = value.to_le_bytes();
            [low, high, 0, 0]
        }
    }
}

#[test]
fn scaling_matches_integer_reference_with_padding_and_clipping() -> Result<(), VideoError> {
    for format in FORMATS {
        for (sw, sh, dw, dh) in [
            (1, 1, 9, 7),
            (7, 5, 3, 2),
            (3, 2, 7, 5),
            (9, 7, 9, 7),
            (63, 48, 79, 57),
        ] {
            let source: Vec<u8> = (0..sw * sh * 3).map(|i| (i * 37) as u8).collect();
            let plan = NearestRgb8::new(sw, sh, dw, dh)?;
            for (x, y) in [(0, 0), (2, 1), (dw - 1, dh - 1), (usize::MAX, usize::MAX)] {
                let bpp = format.bytes_per_pixel();
                let stride = dw * bpp + 11;
                let mut got = vec![123; stride * dh + 17];
                let mut expected = got.clone();
                let mut surface = SurfaceMut::new(&mut got, dw, dh, stride, format)?;
                plan.blit(&source, &mut surface, x, y)?;
                for dy in 0..dh.saturating_sub(y) {
                    for dx in 0..dw.saturating_sub(x) {
                        let src = ((dy * sh / dh) * sw + dx * sw / dw) * 3;
                        let dst = (y + dy) * stride + (x + dx) * bpp;
                        expected[dst..dst + bpp].copy_from_slice(
                            &reference_pixel(&source[src..src + 3], format)[..bpp],
                        );
                    }
                }
                assert_eq!(
                    got, expected,
                    "{format:?}: {sw}x{sh} -> {dw}x{dh} at {x},{y}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn invalid_layout_and_short_source_are_rejected_before_writes() -> Result<(), VideoError> {
    let mut bytes = [77; 32];
    for (w, h, stride) in [
        (0, 1, 4),
        (1, 0, 4),
        (2, 1, 7),
        (1, 2, usize::MAX),
        (usize::MAX, 1, 4),
        (3, 3, 12),
    ] {
        assert!(SurfaceMut::new(&mut bytes, w, h, stride, SurfaceFormat::Bgra8).is_err());
    }
    for (sw, sh, dw, dh) in [
        (0, 1, 1, 1),
        (1, 1, 0, 1),
        (usize::MAX, 2, 1, 1),
        (1, 1, usize::MAX, 2),
    ] {
        assert!(NearestRgb8::new(sw, sh, dw, dh).is_err());
    }
    let plan = NearestRgb8::new(2, 2, 2, 2)?;
    let mut surface = SurfaceMut::new(&mut bytes, 2, 2, 8, SurfaceFormat::Bgra8)?;
    assert!(plan.blit(&[0; 11], &mut surface, 0, 0).is_err());
    assert_eq!(bytes, [77; 32]);
    Ok(())
}

#[test]
fn rectangles_clip_overflowing_extents_without_touching_padding() -> Result<(), VideoError> {
    for format in FORMATS {
        let bpp = format.bytes_per_pixel();
        let stride = 5 * bpp + 3;
        let mut bytes = vec![99; stride * 4 - 3];
        let mut expected = bytes.clone();
        let mut surface = SurfaceMut::new(&mut bytes, 5, 4, stride, format)?;
        surface.fill_rect(2, 1, usize::MAX, usize::MAX, (252, 128, 16));
        surface.fill_rect(usize::MAX, usize::MAX, 7, 7, (0, 0, 0));
        surface.draw_rect(usize::MAX, 0, usize::MAX, 9, 3, (0, 0, 0));
        for y in 1..4 {
            for x in 2..5 {
                let start = y * stride + x * bpp;
                expected[start..start + bpp]
                    .copy_from_slice(&reference_pixel(&[252, 128, 16], format)[..bpp]);
            }
        }
        assert_eq!(bytes, expected);
    }
    Ok(())
}

#[test]
fn outline_respects_zero_sizes_and_excessive_thickness() -> Result<(), VideoError> {
    let mut bytes = [0; 7 * 5 * 3];
    let mut surface = SurfaceMut::new(&mut bytes, 7, 5, 21, SurfaceFormat::Rgb8)?;
    surface.draw_rect(1, 1, 3, 2, usize::MAX, (1, 2, 3));
    surface.draw_rect(0, 0, 0, 5, 3, (99, 99, 99));
    surface.draw_rect(0, 0, 7, 0, 3, (99, 99, 99));
    surface.draw_rect(0, 0, 7, 5, 0, (99, 99, 99));
    for y in 0..5 {
        for x in 0..7 {
            let expected = if (1..4).contains(&x) && (1..3).contains(&y) {
                [1, 2, 3]
            } else {
                [0; 3]
            };
            assert_eq!(&bytes[(y * 7 + x) * 3..(y * 7 + x + 1) * 3], &expected);
        }
    }
    Ok(())
}

#[test]
fn offset_maps_match_wide_integer_formula() {
    for source in [1, 2, 7, 129, 640, 1280, 4093] {
        for destination in [1, 3, 8, 63, 720, 1920] {
            let got = offsets(source, destination, 3);
            for (i, offset) in got.into_iter().enumerate() {
                assert_eq!(
                    offset,
                    (i as u128 * source as u128 / destination as u128) as usize * 3
                );
            }
        }
    }
}
