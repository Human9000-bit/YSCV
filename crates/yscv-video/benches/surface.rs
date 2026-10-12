#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use yscv_video::surface::{NearestRgb8, SurfaceFormat, SurfaceMut};

// Reference workload: cached nearest-neighbor pixel indices, fused RGB8 -> BGRA8.
// Both paths reuse their maps/output; setup and allocation are outside timing.
fn baseline(
    src: &[u8],
    sw: usize,
    columns: &[usize],
    rows: &[usize],
    dst: &mut [u8],
    stride: usize,
) {
    let row_len = columns.len() * 4;
    let source_len = sw * 3;
    for (dy, &sy) in rows.iter().enumerate() {
        let (source_row, dest_row) = (sy * source_len, dy * stride);
        if source_row + source_len > src.len() || dest_row + row_len > dst.len() {
            continue;
        }
        let source = &src[source_row..source_row + source_len];
        let dest = &mut dst[dest_row..dest_row + row_len];
        for (pixel, &sx) in dest.as_chunks_mut::<4>().0.iter_mut().zip(columns) {
            let s = sx * 3;
            *pixel = [source[s + 2], source[s + 1], source[s], 255];
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb8_bgra8");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(20);
    for (sw, sh, dw, dh) in [
        (64, 48, 80, 60),
        (640, 480, 640, 480),
        (640, 480, 720, 576),
        (1280, 720, 640, 480),
        (720, 320, 720, 320),
    ] {
        let id = format!("{sw}x{sh}-{dw}x{dh}");
        let src: Vec<u8> = (0..sw * sh * 3).map(|i| (i * 37) as u8).collect();
        let mut dst = vec![0; dw * dh * 4];
        let mut expected = dst.clone();
        let cols: Vec<_> = (0..dw).map(|x| x * sw / dw).collect();
        let rows: Vec<_> = (0..dh).map(|y| y * sh / dh).collect();
        let plan = NearestRgb8::new(sw, sh, dw, dh).expect("valid benchmark dimensions");
        baseline(&src, sw, &cols, &rows, &mut expected, dw * 4);
        {
            let mut surface = SurfaceMut::new(&mut dst, dw, dh, dw * 4, SurfaceFormat::Bgra8)
                .expect("valid surface");
            plan.blit(&src, &mut surface, 0, 0).expect("valid source");
        }
        assert_eq!(dst, expected);
        group.bench_function(BenchmarkId::new("cached_indices", &id), |b| {
            b.iter(|| {
                baseline(
                    black_box(&src),
                    sw,
                    &cols,
                    &rows,
                    black_box(&mut dst),
                    dw * 4,
                )
            });
        });
        group.bench_function(BenchmarkId::new("surface_plan", &id), |b| {
            let mut surface = SurfaceMut::new(&mut dst, dw, dh, dw * 4, SurfaceFormat::Bgra8)
                .expect("valid surface");
            b.iter(|| {
                plan.blit(black_box(&src), black_box(&mut surface), 0, 0)
                    .expect("valid source")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
