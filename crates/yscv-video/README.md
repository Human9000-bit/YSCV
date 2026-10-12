# yscv-video

H.264, HEVC, and AV1 video decoders, MP4/MKV container parsers, hardware decode, and camera input. Pure Rust, no FFmpeg.

```rust,ignore
use yscv_video::Mp4VideoReader;

let mut reader = Mp4VideoReader::open("video.mp4")?;
while let Some(frame) = reader.next_frame()? {
    // frame.rgb8_data: Vec<u8>, frame.width, frame.height
}
```

## Capabilities

| Feature | Details |
|---------|---------|
| **H.264 decode** | CAVLC, CABAC, I/P/B slices, 4x4/8x8 DCT, weighted prediction |
| **HEVC decode** | CTU quad-tree, branchless CABAC, 8-tap MC, SAO, deblock, weighted prediction, parallel tile/WPP decode, Main 10/12, 4:2:2/4:4:4 |
| **AV1 decode** | OBU parser, intra + inter prediction, 8-tap Lanczos MC, CDEF, deblock |
| **MP4 parse** | Streaming moov-only reader, O(1) memory per frame |
| **MKV parse** | EBML demuxer with frame index |
| **HW decode** | VideoToolbox (macOS), VA-API (Linux), NVDEC, MediaFoundation |
| **Camera** | Live capture via Nokhwa (optional) |
| **Audio** | AAC/ALAC/Opus/Vorbis/MP3/FLAC codec detection |
| **Color** | YUV420→RGB8 with NEON/SSE2 SIMD |
| **Composition** | Checked byte surfaces: RGB8, BGRA8, little-endian RGB565; clipped rectangles; cached nearest-neighbor scaling fused with conversion |
| **Frame broadcast** | One retained frame, independent readers, shared pixel buffers, capture timestamps, bounded waits and close notification |

## Direct composition and frame delivery

`surface::SurfaceMut` borrows a caller-owned byte buffer with an explicit stride
and format. `NearestRgb8` caches sampling offsets and copies repeated rows; blits
allocate nothing. Contiguous RGB8-to-BGRA8 rows use runtime-dispatched SSSE3 or
NEON (AArch64 and ARMv7 with `neon-v7`), with a scalar fallback. Arbitrary scaling
uses a fused scalar gather; no intermediate image or worker threads are needed.
Invalid dimensions, overflow, short strides and truncated input return `VideoError`
before any pixels are written. Padding and clipped-out pixels remain untouched.

```rust
use yscv_video::surface::{NearestRgb8, SurfaceFormat, SurfaceMut};

let rgb = vec![128; 320 * 240 * 3];
let mut output = vec![0; 640 * 480 * 4];
let plan = NearestRgb8::new(320, 240, 640, 480)?; // reuse across frames
let mut surface = SurfaceMut::new(&mut output, 640, 480, 640 * 4, SurfaceFormat::Bgra8)?;
plan.blit(&rgb, &mut surface, 0, 0)?;
surface.draw_rect(10, 20, 80, 60, 2, (255, 0, 0));
# Ok::<(), yscv_video::VideoError>(())
```

`latest_frame::FrameSender<T>` publishes `Arc<T>` payloads with dimensions and a
producer-supplied `Instant`. Each subscriber independently receives the newest
unread frame; slow readers skip intermediate frames without blocking publication
on a full queue. Publishing and receiving allocate no pixel storage. Dimensions
and pixel layout are the producer's responsibility. Dropping all sender handles,
or explicitly closing the channel, wakes readers; the final unread frame can still
be received once. Timeouts retain one deadline across spurious wakeups.

```rust
use std::{sync::Arc, time::{Duration, Instant}};
use yscv_video::latest_frame::FrameSender;

let sender = FrameSender::new();
let mut preview = sender.subscribe();
let mut recorder = sender.subscribe();
sender.publish(Arc::new(vec![0u8; 320 * 240 * 3]), 320, 240, Instant::now())?;
let a = preview.recv_timeout(Duration::from_millis(100))?;
let b = recorder.recv()?;
assert!(Arc::ptr_eq(&a.data, &b.data));
sender.close();
# Ok::<(), yscv_video::latest_frame::FrameChannelError>(())
```

Run `cargo bench -p yscv-video --bench surface` for the cached-index baseline
comparison, and `--bench latest_frame` for publication to two consumers.

## Errors

Every fallible call returns `VideoError`. Its variants carry typed fields
instead of pre-rendered text, so a caller can branch on the kind of failure:

| Variant | Raised when |
|---------|-------------|
| `Truncated`, `OutOfBounds` | input ends before a NAL, OBU, JPEG segment, EBML element or MP4 box is complete |
| `Missing` | a required element never arrived (SPS before a slice, `moov` box, SOF0) |
| `Malformed`, `FieldOutOfRange { field, value }` | a syntax element is invalid or outside its spec range |
| `InvalidDimensions`, `BufferTooSmall`, `FileTooLarge` | sizes that cannot be decoded or held in memory |
| `UnsupportedCodec`, `HwDecode { backend, op, status }` | a hardware backend refused the codec or a call returned a non-zero status |
| `Io { op, path, source }` | file, device, `ioctl` or `mmap` failures, with the OS error as `source` |
| `Camera`, `CameraFallback` | `nokhwa` capture failures (feature `native-camera`) |
| `Unavailable`, `DecodePanicked`, `MavlinkCrcMismatch` | a stream or worker is not ready, a decoder panicked, a MAVLink frame failed its CRC |

## Performance

- H.264 decode: ~4.5x faster than ffmpeg (1080p real camera, Apple M-series) *(measured Apr 2026, pending re-measurement)*
- HEVC decode: ~1.4x faster than ffmpeg (P/B frames, full color) *(measured Apr 2026, pending re-measurement)*
- NEON and x86 SIMD kernels use `#[target_feature]` dispatch through cached
  `yscv-cpu` feature detection, including SSSE3/NEON byte-surface conversion.

## Features

```toml
[features]
videotoolbox = []      # macOS HW decode
vaapi = []             # Linux HW decode
nvdec = []             # NVIDIA HW decode
media-foundation = []  # Windows HW decode
native-camera = []     # Live camera input
```

## Tests

349 default-feature library tests covering decode correctness, container parsing,
surface/reference pixel parity, SIMD tails, clipping, overflow, independent frame
readers, close/wakeup races and timeout deadlines.
