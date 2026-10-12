use std::time::Instant;

use crate::LinearKalman;

const DT_MIN: f32 = 0.005;
const DT_MAX: f32 = 0.200;
const G: f32 = 9.81;
const Z_MIN: f32 = 0.3;

#[derive(Clone, Debug)]
pub struct GuidanceConfig {
    pub kf_p0: f32,
    pub kf_q_pos: f32,
    pub kf_q_vel: f32,
    pub kf_r: f32,
    pub kf_coast_frames: u32,
    pub pn_n: f32,
    pub pn_kp: f32,
    pub pn_vc_scale: f32,
    pub tilt_compensate: bool,
    pub rng_q_z: f32,
    pub rng_q_vz: f32,
    pub rng_q_w0: f32,
    pub rng_r_bbox: f32,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            kf_p0: 1.0,
            kf_q_pos: 0.001,
            kf_q_vel: 0.01,
            kf_r: 0.05,
            kf_coast_frames: 10,
            pn_n: 3.0,
            pn_kp: 0.5,
            pn_vc_scale: 0.1,
            tilt_compensate: true,
            rng_q_z: 0.1,
            rng_q_vz: 1.0,
            rng_q_w0: 0.0001,
            rng_r_bbox: 0.01,
        }
    }
}

/// 2D LOS KF: state `[x, y, vx, vy]`, measurement `[x, y]`.
#[derive(Debug, Clone)]
pub struct LosKf {
    kf: LinearKalman<4, 2>,
    p0: f32,
    q_pos: f32,
    q_vel: f32,
    r: f32,
    last_t: Option<Instant>,
}

impl LosKf {
    pub fn new(cfg: &GuidanceConfig) -> Self {
        let p0 = cfg.kf_p0;
        Self {
            kf: LinearKalman::new([0.0; 4], diag4(p0, p0, p0, p0)),
            p0,
            q_pos: cfg.kf_q_pos,
            q_vel: cfg.kf_q_vel,
            r: cfg.kf_r,
            last_t: None,
        }
    }

    pub fn reset(&mut self) {
        self.kf.x = [0.0; 4];
        self.kf.p = diag4(self.p0, self.p0, self.p0, self.p0);
        self.last_t = None;
    }

    pub fn step(&mut self, err_x: f32, err_y: f32) -> (f32, f32, f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return self.state();
                }
                d.min(DT_MAX)
            }
            None => {
                self.kf.x[0] = err_x;
                self.kf.x[1] = err_y;
                self.last_t = Some(now);
                return (err_x, err_y, 0.0, 0.0);
            }
        };
        self.last_t = Some(now);
        self.predict(dt);
        let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let r = [[self.r, 0.0], [0.0, self.r]];
        self.kf.update(&[err_x, err_y], &h, &r);
        self.state()
    }

    pub fn predict_only(&mut self) -> (f32, f32, f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return self.state();
                }
                d.min(DT_MAX)
            }
            None => return (0.0, 0.0, 0.0, 0.0),
        };
        self.last_t = Some(now);
        self.predict(dt);
        self.state()
    }

    fn predict(&mut self, dt: f32) {
        let f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let q = diag4(self.q_pos, self.q_pos, self.q_vel, self.q_vel);
        self.kf.predict(&f, &q);
    }

    fn state(&self) -> (f32, f32, f32, f32) {
        (self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3])
    }
}

/// 3-state range EKF: `[z, z_dot, w0]`, measurement `h(x) = w0/z`.
#[derive(Debug, Clone)]
pub struct RangeKf {
    x: [f32; 3],
    p: [[f32; 3]; 3],
    q_z: f32,
    q_vz: f32,
    q_w0: f32,
    r_bbox: f32,
    last_t: Option<Instant>,
}

impl RangeKf {
    pub fn new(cfg: &GuidanceConfig) -> Self {
        Self {
            x: [0.0; 3],
            p: [[0.0; 3]; 3],
            q_z: cfg.rng_q_z,
            q_vz: cfg.rng_q_vz,
            q_w0: cfg.rng_q_w0,
            r_bbox: cfg.rng_r_bbox,
            last_t: None,
        }
    }

    pub fn init(&mut self, w_bbox_norm: f32) {
        let z = 1.0 / w_bbox_norm.max(0.01);
        self.x = [z, 0.0, 1.0];
        self.p = [[z * z, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 2.0]];
        self.last_t = None;
    }

    pub fn reset(&mut self) {
        self.x = [0.0; 3];
        self.p = [[0.0; 3]; 3];
        self.last_t = None;
    }

    pub fn step(&mut self, w_bbox_norm: f32, pitch: f32, imu_ax: f32) -> (f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return (self.x[0].max(Z_MIN), self.x[1]);
                }
                d.min(DT_MAX)
            }
            None => {
                self.last_t = Some(now);
                return (self.x[0].max(Z_MIN), self.x[1]);
            }
        };
        self.last_t = Some(now);
        self.predict(dt, pitch, imu_ax);
        if w_bbox_norm > 0.01 {
            self.update(w_bbox_norm);
        }
        self.x[0] = self.x[0].max(Z_MIN);
        self.x[2] = self.x[2].max(0.001);
        (self.x[0], self.x[1])
    }

    pub fn predict_only(&mut self, pitch: f32, imu_ax: f32) -> (f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return (self.x[0].max(Z_MIN), self.x[1]);
                }
                d.min(DT_MAX)
            }
            None => return (self.x[0].max(Z_MIN), self.x[1]),
        };
        self.last_t = Some(now);
        self.predict(dt, pitch, imu_ax);
        self.x[0] = self.x[0].max(Z_MIN);
        (self.x[0], self.x[1])
    }

    fn predict(&mut self, dt: f32, pitch: f32, imu_ax: f32) {
        let az_rel = G * pitch.sin() - imu_ax;
        self.x[0] += self.x[1] * dt;
        self.x[1] += az_rel * dt;

        let p = &mut self.p;
        p[0][0] += dt * (p[0][1] + p[1][0]) + dt * dt * p[1][1] + self.q_z;
        p[0][1] += dt * p[1][1];
        p[0][2] += dt * p[1][2];
        p[1][0] += dt * p[1][1];
        p[1][1] += self.q_vz;
        p[2][0] += dt * p[2][1];
        p[2][2] += self.q_w0;
    }

    fn update(&mut self, w_meas: f32) {
        let z = self.x[0];
        let w0 = self.x[2];
        if z.abs() < 1e-6 {
            return;
        }

        let y = w_meas - w0 / z;
        let h = [-w0 / (z * z), 0.0, 1.0 / z];

        let s = h[0] * h[0] * self.p[0][0]
            + h[0] * h[2] * self.p[0][2]
            + h[2] * h[0] * self.p[2][0]
            + h[2] * h[2] * self.p[2][2]
            + self.r_bbox;

        if s.abs() < 1e-12 {
            return;
        }
        let s_inv = 1.0 / s;

        let k = [
            (self.p[0][0] * h[0] + self.p[0][2] * h[2]) * s_inv,
            (self.p[1][0] * h[0] + self.p[1][2] * h[2]) * s_inv,
            (self.p[2][0] * h[0] + self.p[2][2] * h[2]) * s_inv,
        ];

        self.x[0] += k[0] * y;
        self.x[1] += k[1] * y;
        self.x[2] += k[2] * y;

        let p_old = self.p;
        for i in 0..3 {
            let kh0 = k[i] * h[0];
            let kh2 = k[i] * h[2];
            self.p[i][0] = p_old[i][0] - kh0 * p_old[0][0] - kh2 * p_old[2][0];
            self.p[i][1] = p_old[i][1] - kh0 * p_old[0][1] - kh2 * p_old[2][1];
            self.p[i][2] = p_old[i][2] - kh0 * p_old[0][2] - kh2 * p_old[2][2];
        }
    }
}

pub struct Nav3dOut {
    pub err_x: f32,
    pub err_y: f32,
    pub rate_x: f32,
    pub rate_y: f32,
    pub v_closing: f32,
    pub range: f32,
}

/// Combined 2D LOS + range EKF.
pub struct Nav3d {
    pub los: LosKf,
    pub rng: RangeKf,
    pub initialized: bool,
}

impl Nav3d {
    pub fn new(cfg: &GuidanceConfig) -> Self {
        Self {
            los: LosKf::new(cfg),
            rng: RangeKf::new(cfg),
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        self.los.reset();
        self.rng.reset();
        self.initialized = false;
    }

    pub fn step(
        &mut self,
        err_x: f32,
        err_y: f32,
        w_bbox_norm: f32,
        pitch: f32,
        imu_ax: f32,
    ) -> Nav3dOut {
        if !self.initialized && w_bbox_norm > 0.01 {
            self.rng.init(w_bbox_norm);
            self.initialized = true;
        }

        let (ex, ey, rx, ry) = self.los.step(err_x, err_y);
        let (z, z_dot) = if self.initialized {
            self.rng.step(w_bbox_norm, pitch, imu_ax)
        } else {
            (1.0, 0.0)
        };

        Nav3dOut {
            err_x: ex,
            err_y: ey,
            rate_x: rx,
            rate_y: ry,
            v_closing: (-z_dot).max(0.0),
            range: z,
        }
    }

    pub fn predict_only(&mut self, pitch: f32, imu_ax: f32) -> Nav3dOut {
        let (ex, ey, rx, ry) = self.los.predict_only();
        let (z, z_dot) = if self.initialized {
            self.rng.predict_only(pitch, imu_ax)
        } else {
            (1.0, 0.0)
        };

        Nav3dOut {
            err_x: ex,
            err_y: ey,
            rate_x: rx,
            rate_y: ry,
            v_closing: (-z_dot).max(0.0),
            range: z,
        }
    }
}

pub fn compensate_tilt(err_x: f32, err_y: f32, roll: f32, _pitch: f32) -> (f32, f32) {
    let (cr, sr) = (roll.cos(), roll.sin());
    (cr * err_x + sr * err_y, -sr * err_x + cr * err_y)
}

pub fn pn_sticks(kp: f32, n: f32, err_x: f32, err_y: f32, rate_x: f32, rate_y: f32) -> (f32, f32) {
    let sx = (kp * err_x + n * rate_x).clamp(-1.0, 1.0);
    let sy = (kp * err_y + n * rate_y).clamp(-1.0, 1.0);
    (sx, sy)
}

pub fn pn3d_sticks(kp: f32, n: f32, vc_scale: f32, out: &Nav3dOut) -> (f32, f32) {
    let vc_factor = 1.0 + out.v_closing * vc_scale;
    let sx = (kp * out.err_x + n * out.rate_x * vc_factor).clamp(-1.0, 1.0);
    let sy = (kp * out.err_y + n * out.rate_y * vc_factor).clamp(-1.0, 1.0);
    (sx, sy)
}

fn diag4(a: f32, b: f32, c: f32, d: f32) -> [[f32; 4]; 4] {
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = a;
    m[1][1] = b;
    m[2][2] = c;
    m[3][3] = d;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.033;

    #[test]
    fn los_first_step_passthrough() {
        let cfg = GuidanceConfig::default();
        let mut kf = LosKf::new(&cfg);
        let (ex, ey, rx, ry) = kf.step(0.3, -0.5);
        assert!((ex - 0.3).abs() < 1e-6);
        assert!((ey + 0.5).abs() < 1e-6);
        assert_eq!((rx, ry), (0.0, 0.0));
    }

    #[test]
    fn los_converges_to_stationary() {
        let cfg = GuidanceConfig::default();
        let mut kf = LosKf::new(&cfg);
        kf.kf.x = [0.0; 4];
        kf.kf.p = diag4(1.0, 1.0, 1.0, 1.0);
        for _ in 0..80 {
            kf.predict(DT);
            let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
            let r = [[kf.r, 0.0], [0.0, kf.r]];
            kf.kf.update(&[0.3, -0.2], &h, &r);
        }
        assert!((kf.kf.x[0] - 0.3).abs() < 0.01);
        assert!((kf.kf.x[1] + 0.2).abs() < 0.01);
        assert!(kf.kf.x[2].abs() < 0.05);
        assert!(kf.kf.x[3].abs() < 0.05);
    }

    #[test]
    fn los_tracks_constant_velocity() {
        let cfg = GuidanceConfig::default();
        let mut kf = LosKf::new(&cfg);
        kf.kf.x = [0.0; 4];
        kf.kf.p = diag4(1.0, 1.0, 1.0, 1.0);
        let (vx, vy) = (0.5, -0.3);
        for i in 1..=100 {
            let t = i as f32 * DT;
            kf.predict(DT);
            let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
            let r = [[kf.r, 0.0], [0.0, kf.r]];
            kf.kf.update(&[vx * t, vy * t], &h, &r);
        }
        assert!((kf.kf.x[2] - vx).abs() < 0.1);
        assert!((kf.kf.x[3] - vy).abs() < 0.1);
    }

    #[test]
    fn pn_sticks_clamp() {
        let (sx, sy) = pn_sticks(10.0, 5.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(sx, 1.0);
        assert_eq!(sy, 1.0);
    }

    #[test]
    fn pn_zero_in_zero_out() {
        assert_eq!(pn_sticks(0.5, 3.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn compensate_tilt_zero_roll() {
        let (x, y) = compensate_tilt(0.5, 0.3, 0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-6);
        assert!((y - 0.3).abs() < 1e-6);
    }

    #[test]
    fn compensate_tilt_preserves_magnitude() {
        let roll = 25.0f32.to_radians();
        let (cx, cy) = compensate_tilt(0.3, 0.4, roll, 0.0);
        let orig = (0.3f32 * 0.3 + 0.4 * 0.4).sqrt();
        let comp = (cx * cx + cy * cy).sqrt();
        assert!((comp - orig).abs() < 1e-5);
    }

    #[test]
    fn range_kf_init() {
        let cfg = GuidanceConfig::default();
        let mut rng = RangeKf::new(&cfg);
        rng.init(0.1);
        assert!((rng.x[0] - 10.0).abs() < 1e-3);
        assert!((rng.x[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn range_stable_on_constant_bbox() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.1);
        let z0 = rk.x[0];
        for _ in 0..50 {
            rk.predict(DT, 0.0, 0.0);
            rk.update(0.1);
        }
        assert!((rk.x[0] - z0).abs() < z0 * 0.2);
        assert!(rk.x[1].abs() < 1.0);
    }

    #[test]
    fn range_approach_gives_negative_zdot() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.1);
        for i in 0..60 {
            let w = 0.1 + 0.002 * i as f32;
            rk.predict(DT, 0.0, 0.0);
            rk.update(w);
        }
        assert!(rk.x[1] < 0.0, "approach -> negative z_dot: {}", rk.x[1]);
    }

    #[test]
    fn range_recede_gives_positive_zdot() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.2);
        for i in 0..60 {
            let w = (0.2 - 0.002 * i as f32).max(0.02);
            rk.predict(DT, 0.0, 0.0);
            rk.update(w);
        }
        assert!(rk.x[1] > 0.0, "recede -> positive z_dot: {}", rk.x[1]);
    }

    #[test]
    fn range_z_floor_holds() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.5);
        for _ in 0..50 {
            rk.predict(DT, -0.5, 0.0);
            rk.update(5.0);
        }
        assert!(rk.x[0] >= Z_MIN);
    }

    #[test]
    fn range_p_stays_symmetric() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.1);
        for i in 0..30 {
            rk.predict(DT, 0.05, 0.0);
            rk.update(0.1 + 0.002 * i as f32);
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let (a, b) = (rk.p[i][j], rk.p[j][i]);
                assert!((a - b).abs() < 1e-4, "P[{i}][{j}]={a} != P[{j}][{i}]={b}");
            }
        }
    }

    #[test]
    fn range_accel_affects_velocity() {
        let cfg = GuidanceConfig::default();
        let mut still = RangeKf::new(&cfg);
        still.init(0.1);
        let mut accel = RangeKf::new(&cfg);
        accel.init(0.1);
        for _ in 0..30 {
            still.predict(DT, 0.0, 0.0);
            still.update(0.1);
            accel.predict(DT, 0.3, 0.0);
            accel.update(0.1);
        }
        assert!((accel.x[1] - still.x[1]).abs() > 0.01);
    }

    #[test]
    fn pn3d_closing_does_not_affect_kp_term() {
        let a = Nav3dOut {
            err_x: 0.3,
            err_y: 0.0,
            rate_x: 0.0,
            rate_y: 0.0,
            v_closing: 0.0,
            range: 5.0,
        };
        let b = Nav3dOut {
            v_closing: 10.0,
            ..a
        };
        let (sa, _) = pn3d_sticks(0.5, 3.0, 0.1, &a);
        let (sb, _) = pn3d_sticks(0.5, 3.0, 0.1, &b);
        assert!((sa - sb).abs() < 1e-6);
    }
}
