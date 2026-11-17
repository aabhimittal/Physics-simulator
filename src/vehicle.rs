//! Cars, the way most game engines build them (a "raycast vehicle"): the body is an
//! ordinary rigid body in the `World`, and each wheel is a ray down from the body
//! carrying a spring-damper suspension and a tyre model. The wheels are not bodies:
//! rolling contact is millimetres deep and changes every step, which rigid contact
//! handles badly, so all the grip comes from the tyre model instead.
//!
//! Tyre forces use Pacejka's "Magic Formula": grip grows with slip, peaks (about 13%
//! wheel slip, or 8° of sideways slip) and then falls to about 80% of the peak, so a
//! locked or spinning wheel grips less than one rolling just below the peak. That is
//! why the brakes have ABS. Both directions together are capped by the friction
//! circle, |F| ≤ μ·load.

use crate::{Body, Quat, Vec3};

const RADIUS: f64 = 0.33;
/// Suspension length (mount to wheel centre) with no load, and how far it can compress.
const REST: f64 = 0.35;
const TRAVEL: f64 = 0.2;
/// Per wheel: about 1.5 Hz bounce on a quarter of a 1200 kg car, damped at 30% of critical.
const SPRING: f64 = 26_000.0;
const DAMPER: f64 = 1_700.0;
/// Hard spring once the suspension runs out of travel.
const BUMP_STOP: f64 = 300_000.0;
/// How far each spring compresses under a quarter of the car's 1200 kg.
const SAG: f64 = 1200.0 * 9.81 / 4.0 / SPRING;
const WHEEL_INERTIA: f64 = 1.2;
/// Tyre-road friction coefficient (dry asphalt).
const GRIP: f64 = 1.0;
/// Total torque at the driven (rear) wheels in first gear, and the engine's power.
/// Above about 67 km/h (power / full-torque force) the power limit takes over, as with a real engine and gearbox.
const ENGINE_TORQUE: f64 = 1_600.0;
const ENGINE_POWER: f64 = 90_000.0;
/// Brake torque per wheel at full pedal, rear wheels 60% of it (front bias, as on real cars).
const BRAKE_TORQUE: f64 = 1_800.0;
/// ABS lets a wheel's brake off while it slips past the tyre's grip peak.
const ABS_SLIP: f64 = 0.15;
/// Magic Formula stiffness and shape: along the wheel, then sideways. With C = 1.4 a fully
/// sliding tyre keeps sin(1.4·π/2) = 81% of its peak grip.
const LONG: (f64, f64) = (16.0, 1.4);
const LAT: (f64, f64) = (14.0, 1.4);
const ROLLING_RESISTANCE: f64 = 0.015;
/// Air drag: ½·ρ·Cd·A for a small car.
const DRAG: f64 = 0.5 * 1.2 * 0.65;
/// Below this speed (m/s) slip is measured against it instead, so slip stays finite at standstill.
const SLOW: f64 = 1.0;
pub const WHEELBASE: f64 = 2.7;
pub const MAX_STEER: f64 = 0.6;

/// Pacejka's Magic Formula with E = 0: stiffness B, shape C, peak D.
fn magic(slip: f64, b: f64, c: f64, d: f64) -> f64 {
    d * (c * (b * slip).atan()).sin()
}

pub struct Wheel {
    /// Top of the suspension, in the body frame.
    pub mount: Vec3,
    pub front: bool,
    /// Spin rate (rad/s) and how far the wheel has turned (rad, for drawing).
    pub spin: f64,
    pub roll: f64,
    /// Suspension length now, and the steering angle of this wheel.
    pub length: f64,
    pub steer: f64,
    /// Normal force on the tyre (N); 0 in the air.
    pub load: f64,
    /// Wheel slip ratio and sideways slip angle (rad).
    pub slip: (f64, f64),
}

pub struct Car {
    /// Index of the car's body in the `World`.
    pub body: usize,
    pub wheels: Vec<Wheel>,
    /// Driver inputs: throttle 0..1 (negative drives backwards), brake 0..1, steering (rad).
    pub throttle: f64,
    pub brake: f64,
    pub steer: f64,
    /// When set, the car drives itself round a circle of this radius about the origin.
    pub autopilot: Option<f64>,
    pub time: f64,
    /// Acceleration of the body (m/s²), for display.
    pub accel: Vec3,
    last_vel: Vec3,
}

impl Car {
    /// A 1200 kg, rear-wheel-drive car on `body`, whose box should be about 4.2 x 0.7 x 1.4 m.
    pub fn new(body: usize) -> Car {
        let wheel = |x: f64, z: f64| Wheel {
            mount: Vec3::new(x, -0.1, z),
            front: x > 0.0,
            spin: 0.0,
            roll: 0.0,
            length: REST - SAG,
            steer: 0.0,
            load: 0.0,
            slip: (0.0, 0.0),
        };
        let x = WHEELBASE / 2.0;
        Car {
            body,
            wheels: vec![
                wheel(x, 0.8),
                wheel(x, -0.8),
                wheel(-x, 0.8),
                wheel(-x, -0.8),
            ],
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            autopilot: None,
            time: 0.0,
            accel: Vec3::ZERO,
            last_vel: Vec3::ZERO,
        }
    }

    /// Height of the body's centre above flat ground at rest: each spring carries a quarter of the weight.
    pub const RIDE_HEIGHT: f64 = RADIUS + REST - SAG + 0.1;

    /// Applies one substep of suspension, tyre, drive, brake and drag forces to `b`.
    pub fn update(&mut self, b: &mut Body, h: f64) {
        self.time += h;
        if let Some(radius) = self.autopilot {
            self.drive_round(b, radius);
        }
        b.wake(); // a parked car must still respond to its driver
        let speed = b.vel.length();
        b.kick(Vec3::ZERO, b.vel * (-DRAG * speed * h));
        let down = b.rot * Vec3::NEG_Y;
        for w in &mut self.wheels {
            // Ackermann steering: each front wheel points at the same turning centre,
            // so the inner wheel turns more than the outer one.
            let t = self.steer.tan();
            w.steer = if w.front {
                (WHEELBASE * t).atan2(WHEELBASE + w.mount.z * t)
            } else {
                0.0
            };
            // shortcut: the ray only meets flat ground at y = 0; ray against the world's bodies for kerbs and ramps.
            let p = b.pos + b.rot * w.mount;
            let reach = if down.y < -0.2 {
                p.y / -down.y
            } else {
                f64::INFINITY
            };
            let length = (reach - RADIUS).min(REST);
            let rate = (w.length - length) / h; // compression speed
            w.length = length;
            let drive = if w.front {
                0.0
            } else {
                // Open differential: each rear wheel gets half.
                self.throttle * ENGINE_TORQUE.min(ENGINE_POWER / w.spin.abs().max(1.0)) / 2.0
            };
            let mut spin = w.spin;
            let mut slip = (0.0, 0.0);
            let compression = REST - length;
            w.load = if compression > 0.0 {
                let stop = (compression - TRAVEL).max(0.0) * BUMP_STOP;
                (SPRING * compression + DAMPER * rate + stop).max(0.0)
            } else {
                0.0
            };
            if w.load > 0.0 {
                let contact = p + down * reach;
                let v = b.point_vel(contact - b.pos);
                let heading = (b.rot * Quat::from_rotation_y(w.steer) * Vec3::X)
                    .with_y(0.0)
                    .normalize_or_zero();
                let side = heading.cross(Vec3::Y);
                let (along, across) = (v.dot(heading), v.dot(side));
                let scale = along.abs().max(SLOW);
                let peak = GRIP * w.load;
                // The wheel's spin is stiff against the road (a light wheel and a grippy tyre),
                // so it is stepped implicitly: the tyre force is linearised in the new spin.
                // Explicitly it would oscillate and blow up at this time step.
                let ratio = (spin * RADIUS - along) / scale;
                let stiffness = LONG.0 * LONG.1 * peak * RADIUS / scale; // d(force)/d(spin) at small slip
                let resist = ROLLING_RESISTANCE * w.load * RADIUS * spin.signum();
                let pull = magic(ratio, LONG.0, LONG.1, peak);
                spin +=
                    h * (drive - resist - RADIUS * pull) / (WHEEL_INERTIA + h * RADIUS * stiffness);
                slip = ((spin * RADIUS - along) / scale, across.atan2(scale));
                let long = magic(slip.0, LONG.0, LONG.1, peak);
                let lat = -magic(slip.1, LAT.0, LAT.1, peak);
                // Friction circle: both directions share the same grip.
                let tyre = (heading * long + side * lat).clamp_length_max(peak);
                // Applied at the contact patch, not the hub: the difference is the drive and
                // brake torque, which reacts on the body through the engine and brake mounts.
                b.kick(contact - b.pos, (Vec3::Y * w.load + tyre) * h);
            } else {
                spin += h * drive / WHEEL_INERTIA;
            }
            // Brakes slow the wheel toward a stop but never past it: a locked wheel skids.
            let bias = if w.front { 1.0 } else { 0.6 };
            let bias = if slip.0 < -ABS_SLIP { 0.0 } else { bias };
            let stop = self.brake * BRAKE_TORQUE * bias * h / WHEEL_INERTIA;
            w.spin = spin.signum() * (spin.abs() - stop).max(0.0);
            w.roll += w.spin * h;
            w.slip = slip;
        }
        // Smoothed over about 0.1 s, so the readout shows the motion rather than the suspension buzz.
        let a = (b.vel - self.last_vel) / h;
        self.accel += (a - self.accel) * (h / 0.1).min(1.0);
        self.last_vel = b.vel;
    }

    /// Autopilot: pure-pursuit steering toward a point a little way ahead on the circle,
    /// and a speed that alternates between 15 and 8 m/s every 8 s to show acceleration and braking.
    fn drive_round(&mut self, b: &Body, radius: f64) {
        let ahead = 6.0 + 0.6 * b.vel.length();
        let angle = b.pos.z.atan2(b.pos.x) - ahead / radius; // clockwise seen from above
        let target = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
        let local = b.rot.inverse() * (target - b.pos);
        let curvature = -2.0 * local.z / local.length_squared();
        self.steer = (WHEELBASE * curvature).atan().clamp(-MAX_STEER, MAX_STEER);
        let wanted = if (self.time / 8.0) as u64 % 2 == 0 {
            15.0
        } else {
            8.0
        };
        let forward = b.vel.dot(b.rot * Vec3::X);
        let push = (0.5 * (wanted - forward)).clamp(-1.0, 1.0);
        (self.throttle, self.brake) = (push.max(0.0), (-push).max(0.0));
    }

    /// Driver input from the keyboard; turns the autopilot off.
    pub fn drive(&mut self, b: &Body, throttle: f64, steer: f64) {
        self.autopilot = None;
        let forward = b.vel.dot(b.rot * Vec3::X);
        // Keys are all or nothing, so at speed full lock is cut to what the tyres can use: the
        // angle for a 1 g turn plus the slip angle of peak grip. Most driving games do the same.
        let usable = WHEELBASE * 9.81 / forward.powi(2).max(1e-9) + 0.08;
        self.steer = steer.clamp(-1.0, 1.0) * MAX_STEER.min(usable);
        // Pressing against the direction of travel brakes; once nearly stopped it drives that way.
        if throttle * forward < 0.0 && forward.abs() > 0.5 {
            (self.throttle, self.brake) = (0.0, throttle.abs().min(1.0));
        } else {
            (self.throttle, self.brake) = (throttle.clamp(-1.0, 1.0), 0.0);
        }
    }

    /// World centre and orientation of each wheel, for drawing. The wheel's axle is its local y.
    pub fn wheel_poses(&self, b: &Body) -> Vec<(Vec3, Quat)> {
        self.wheels
            .iter()
            .map(|w| {
                let centre = b.pos + b.rot * (w.mount + Vec3::NEG_Y * w.length);
                let rot = b.rot
                    * Quat::from_rotation_y(w.steer)
                    * Quat::from_rotation_z(-w.roll)
                    * Quat::from_rotation_x(std::f64::consts::FRAC_PI_2);
                (centre, rot)
            })
            .collect()
    }

    /// Body roll angle (rad): how far the body leans sideways.
    pub fn roll(&self, b: &Body) -> f64 {
        (b.rot.inverse() * Vec3::Y).z.asin()
    }

    pub const WHEEL_SIZE: (f64, f64) = (RADIUS, 0.11);
}
