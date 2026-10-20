# yscv-track

Multi-object tracking: DeepSORT, ByteTrack, Kalman filter, Hungarian assignment, and Re-ID.

```rust,ignore
use yscv_track::*;

let mut tracker = Tracker::new(TrackerConfig::default());
loop {
    let detections = detect_objects(&frame);
    let tracked = tracker.update(&detections);
    for t in &tracked {
        println!("Track {} at ({}, {})", t.track_id, t.x, t.y);
    }
}
```

## Algorithms

| Algorithm | Use case |
|-----------|----------|
| **DeepSORT** | Appearance + motion, best for re-identification |
| **ByteTrack** | High-speed temporal association, no appearance model |
| **Kalman Filter** | Motion prediction with constant velocity model |
| **Hungarian** | Optimal assignment between tracks and detections |

## Re-ID

- Color histogram features (no model needed)
- Cosine similarity matching
- VP-Tree approximate nearest neighbor
- Feature gallery with FIFO eviction (max 100 per track)

## Guidance

Vision-based guidance laws over a tracker box, in two flavours.

**Sticks (`LosKf`, `Nav3d`)** — screen-space: a 4-state KF smooths the
normalised pixel error, `Nav3d` adds a 3-state range EKF over the bbox width,
and PN produces a pair of stick signals in [-1, 1].

**NED (`NedGuidance`)** — world-space: the pixel is deprojected through
camera intrinsics into a ray, rotated into NED by the mount pitch and the
body attitude, and filtered by a 6-state KF `[los, los_rate]`. The law is
pure proportional navigation `a = N·Vc·(Ω × los)` — an acceleration in m/s²,
perpendicular to the LOS, with the navigation constant adaptive in [2, 5] and
the closing velocity from the range EKF. No intercept point is computed: PN
nulls the LOS rotation, which is itself the lead. Closer than 4 m a terminal
PID takes over; acceleration clamps are asymmetric vertically (NED z is
down). The law's gains are module constants; the per-deployment input is the
camera calibration (`Cam`) beside `GuidanceConfig`.

## Tests

97 tests covering track lifecycle, assignment, Kalman prediction, Re-ID
matching, and the guidance laws (LOS/range filters, PN geometry and lead
direction, terminal-PID handover, clamps, coasting, reset).
