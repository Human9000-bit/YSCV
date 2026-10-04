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
- 21 named SIMD functions (8 NEON + 11 SSE2 + 2 AVX2) with
  `#[target_feature]` runtime dispatch through cached `yscv-cpu`
  feature detection

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

220 tests covering decode correctness, container parsing, edge cases.
