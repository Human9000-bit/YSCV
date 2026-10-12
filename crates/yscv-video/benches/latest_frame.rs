#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};
use yscv_video::latest_frame::FrameSender;

fn bench(c: &mut Criterion) {
    let sender = FrameSender::new();
    let mut first = sender.subscribe();
    let mut second = sender.subscribe();
    let pixels = Arc::new(vec![0; 640 * 480 * 3]);
    let captured_at = Instant::now();
    c.bench_function("latest_frame/publish_two_readers", |b| {
        b.iter(|| {
            sender
                .publish(black_box(pixels.clone()), 640, 480, captured_at)
                .expect("open channel");
            black_box(first.recv_timeout(Duration::ZERO).expect("new frame"));
            black_box(second.recv_timeout(Duration::ZERO).expect("new frame"));
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
