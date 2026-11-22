//! Cars, the way most game engines build them (a "raycast vehicle"): the body is an
//! ordinary rigid body in the `World`, and each wheel is a tyre-sized sphere swept down from
//! the body (a shape cast, so it rolls over edges rather than snapping onto them), carrying a
//! spring-damper suspension and a tyre model. The wheels are not bodies:
//! rolling contact is millimetres deep and changes every step, which rigid contact
//! handles badly, so all the grip comes from the tyre model instead.
//!
//! Tyre forces use Pacejka's "Magic Formula": grip grows with slip, peaks (about 13%
//! wheel slip, or 8° of sideways slip) and then falls to about 80% of the peak, so a
//! locked or spinning wheel grips less than one rolling just below the peak. That is
//! why the brakes have ABS. Both directions together are capped by the friction
//! circle, |F| ≤ μ·load.

use crate::{Body, Quat, Vec3, sphere_cast};

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
/// Engine torque (N·m) against speed (rpm): 175 N·m at 4500 rpm, about 90 kW at 6000.
/// Between points it is linear; past the last one the rev limiter cuts the fuel.
const TORQUE_CURVE: [(f64, f64); 5] = [
    (1000.0, 120.0),
    (2500.0, 160.0),
    (4500.0, 175.0),
    (6000.0, 150.0),
    (6800.0, 130.0),
];
/// A five-speed gearbox, its reverse gear, the final drive and the drivetrain's efficiency.
const GEARS: [f64; 5] = [3.6, 2.2, 1.5, 1.15, 0.9];
const REVERSE: f64 = 3.3;
const FINAL_DRIVE: f64 = 3.9;
const DRIVETRAIN: f64 = 0.9;
/// Automatic shifting (rpm), from no throttle to full: up at 2000 rising to near the red
/// line, down at 1100 rising to 2500 (each up-shift lands above the down-shift point, so it
/// never hunts between gears). Drive is cut for SHIFT_TIME while a gear changes.
const SHIFT_UP_RPM: (f64, f64) = (2000.0, 6200.0);
const SHIFT_DOWN_RPM: (f64, f64) = (1100.0, 2500.0);
const SHIFT_TIME: f64 = 0.25;
/// Below this the clutch slips, so the engine never stalls (rpm).
const IDLE_RPM: f64 = 1000.0;
/// Anti-roll bars: extra spring force per metre of difference between the two wheels on an
/// axle. They resist the body leaning without stiffening the ride over bumps that hit both.
const ANTI_ROLL: [f64; 2] = [20_000.0, 10_000.0];
/// Brake torque per wheel at full pedal, rear wheels 60% of it (front bias, as on real cars).
const BRAKE_TORQUE: f64 = 1_800.0;
/// ABS lets a wheel's brake off while it slips past the tyre's grip peak.
const ABS_SLIP: f64 = 0.15;
/// Magic Formula stiffness and shape: along the wheel, then sideways. With C = 1.4 a fully
/// sliding tyre keeps sin(1.4·π/2) = 81% of its peak grip.
const LONG: (f64, f64) = (16.0, 1.4);
const LAT: (f64, f64) = (14.0, 1.4);
/// Tyre load sensitivity: cornering stiffness grows as load^LOAD_SENSITIVITY about the
/// average load per wheel. Real tyres measure about 0.5 to 0.8; 0.5 matches the Civic's
/// understeer in comma2k19 (it also stands in for steering compliance, not modelled).
const NOMINAL_LOAD: f64 = 1200.0 * 9.81 / 4.0;
const LOAD_SENSITIVITY: f64 = 0.5;
/// Share of the weight on the front wheels (a 2016 Honda Civic's).
const FRONT_WEIGHT: f64 = 0.61;
/// How much the car understeers (s²/m²): it turns 1 / (1 + UNDERSTEER·v²) as tightly as its
/// wheels point. Measured on this model and on the Civic in comma2k19 (tests/vehicle.rs).
const UNDERSTEER: f64 = 6.0e-4;
/// Rolling resistance (force per unit load) and air drag (½·ρ·Cd·A), from the EPA's
/// coast-down road load for the 2016 Civic 1.5 L CVT (tests/vehicle.rs); the rolling figure
/// includes the drivetrain's friction, as a coast-down in neutral does.
const ROLLING_RESISTANCE: f64 = 0.0094;
const DRAG: f64 = 0.5 * 1.2 * 0.675;
/// Below this speed (m/s) slip is measured against it instead, so slip stays finite at standstill.
const SLOW: f64 = 1.0;
pub const WHEELBASE: f64 = 2.7;
/// Firm but ordinary braking (0.6 g) when stopping for someone; the tyres can give about 1 g.
const YIELD_DECELERATION: f64 = 6.0;
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
    /// The body the tyre is on (0 is the ground), if it touches anything.
    pub ground: Option<usize>,
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
    /// Where a person on foot is and how fast they are moving, if the autopilot should watch
    /// for one; and whether it is slowing down for them right now.
    pub pedestrian: Option<(Vec3, Vec3)>,
    pub yielding: bool,
    /// The brakes have failed: the pedal does nothing, and only drag and rolling resistance
    /// slow the car.
    pub brakes_failed: bool,
    pub time: f64,
    /// Acceleration of the body (m/s²), for display.
    pub accel: Vec3,
    last_vel: Vec3,
    /// Gear (1 to 5, or -1 for reverse), engine speed (rpm), and time left in a gear change.
    pub gear: i32,
    pub rpm: f64,
    shifting: f64,
}

impl Car {
    /// A 1200 kg, front-wheel-drive car on `body`, whose box should be about 4.2 x 0.7 x 1.4 m,
    /// with 61% of its weight on the front wheels.
    pub fn new(body: usize) -> Car {
        let wheel = |x: f64, z: f64| Wheel {
            mount: Vec3::new(x, -0.1, z),
            front: x > 0.0,
            spin: 0.0,
            roll: 0.0,
            length: REST - SAG,
            steer: 0.0,
            load: 0.0,
            ground: None,
            slip: (0.0, 0.0),
        };
        // The centre of mass sits FRONT_WEIGHT of the way toward the front axle's load.
        let (front, rear) = (WHEELBASE * (1.0 - FRONT_WEIGHT), -WHEELBASE * FRONT_WEIGHT);
        Car {
            body,
            wheels: vec![
                wheel(front, 0.8),
                wheel(front, -0.8),
                wheel(rear, 0.8),
                wheel(rear, -0.8),
            ],
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            autopilot: None,
            pedestrian: None,
            yielding: false,
            brakes_failed: false,
            time: 0.0,
            accel: Vec3::ZERO,
            last_vel: Vec3::ZERO,
            gear: 1,
            rpm: IDLE_RPM,
            shifting: 0.0,
        }
    }

    /// Height of the body's centre above flat ground at rest: each spring carries a quarter of the weight.
    pub const RIDE_HEIGHT: f64 = RADIUS + REST - SAG + 0.1;

    /// Applies one substep of suspension, tyre, drive, brake and drag forces to the car's
    /// body, and the tyres' reactions to whatever they stand on.
    pub fn update(&mut self, bodies: &mut [Body], h: f64) {
        self.time += h;
        let me = self.body;
        if let Some(radius) = self.autopilot {
            self.drive_round(&bodies[me], radius);
        }
        bodies[me].wake(); // a parked car must still respond to its driver
        let b = &bodies[me];
        let mut impulses = vec![(me, b.pos, b.vel * (-DRAG * b.vel.length() * h))];
        let down = b.rot * Vec3::NEG_Y;
        // Each wheel's sweep finds the nearest surface below it on any body: ground, kerb,
        // ramp or crate. First every suspension length, as the anti-roll bars need both
        // wheels of an axle; then the forces.
        let hits: Vec<_> = self
            .wheels
            .iter()
            .map(|w| sphere_cast(bodies, b.pos + b.rot * w.mount, down, RADIUS, REST, me))
            .collect();
        let lengths: Vec<f64> = hits
            .iter()
            .map(|hit| hit.map_or(REST, |hit| hit.distance.min(REST)))
            .collect();
        let drive = self.drive_torque(h);
        for (k, w) in self.wheels.iter_mut().enumerate() {
            // Ackermann steering: each front wheel points at the same turning centre,
            // so the inner wheel turns more than the outer one.
            let t = self.steer.tan();
            w.steer = if w.front {
                (WHEELBASE * t).atan2(WHEELBASE + w.mount.z * t)
            } else {
                0.0
            };
            let length = lengths[k];
            let rate = (w.length - length) / h; // compression speed
            w.length = length;
            let partner = REST - lengths[k ^ 1]; // the other wheel on the same axle
            let drive = if w.front { drive / 2.0 } else { 0.0 }; // open differential
            let mut spin = w.spin;
            let mut slip = (0.0, 0.0);
            let compression = REST - length;
            w.ground = hits[k].map(|hit| hit.body);
            w.load = match hits[k] {
                Some(_) if compression > 0.0 => {
                    let stop = (compression - TRAVEL).max(0.0) * BUMP_STOP;
                    let bar = ANTI_ROLL[k / 2] * (compression - partner);
                    // Springs and dampers sized to each axle's share, so the car sits level.
                    let share = 2.0
                        * if w.front {
                            FRONT_WEIGHT
                        } else {
                            1.0 - FRONT_WEIGHT
                        };
                    (share * (SPRING * compression + DAMPER * rate) + stop + bar).max(0.0)
                }
                _ => 0.0,
            };
            if let (Some(hit), true) = (hits[k], w.load > 0.0) {
                let p = b.pos + b.rot * w.mount;
                // Where the tyre touches: from the wheel centre against the surface normal.
                let n = hit.normal;
                let contact = p + down * hit.distance - n * RADIUS;
                // Relative to the surface, which may itself be moving (a crate).
                let under = &bodies[hit.body];
                let v = b.point_vel(contact - b.pos) - under.point_vel(contact - under.pos);
                let heading = b.rot * Quat::from_rotation_y(w.steer) * Vec3::X;
                let heading = (heading - n * heading.dot(n)).normalize_or_zero();
                let side = heading.cross(n);
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
                // Cornering stiffness grows more slowly than load, so the heavier front axle
                // grips relatively less: the main source of a front-heavy car's understeer.
                let soften = (NOMINAL_LOAD / w.load).powf(1.0 - LOAD_SENSITIVITY);
                let lat = -magic(slip.1, LAT.0 * soften, LAT.1, peak);
                // Friction circle: both directions share the same grip.
                let tyre = (heading * long + side * lat).clamp_length_max(peak);
                // Applied at the contact patch, not the hub: the difference is the drive and
                // brake torque, which reacts on the body through the engine and brake mounts.
                let force = (n * w.load + tyre) * h;
                impulses.push((me, contact, force));
                impulses.push((hit.body, contact, -force));
            } else {
                spin += h * drive / WHEEL_INERTIA;
            }
            // Brakes slow the wheel toward a stop but never past it: a locked wheel skids.
            let bias = if w.front { 1.0 } else { 0.6 };
            let bias = if slip.0 < -ABS_SLIP { 0.0 } else { bias };
            let pedal = if self.brakes_failed { 0.0 } else { self.brake };
            let stop = pedal * BRAKE_TORQUE * bias * h / WHEEL_INERTIA;
            w.spin = spin.signum() * (spin.abs() - stop).max(0.0);
            w.roll += w.spin * h;
            w.slip = slip;
        }
        for (i, at, impulse) in impulses {
            let body = &mut bodies[i];
            if i != me && impulse.length_squared() > 0.0 {
                body.wake(); // a crate under a wheel must feel it
            }
            let r = at - body.pos;
            body.kick(r, impulse);
        }
        // Smoothed over about 0.1 s, so the readout shows the motion rather than the suspension buzz.
        let b = &bodies[me];
        let a = (b.vel - self.last_vel) / h;
        self.accel += (a - self.accel) * (h / 0.1).min(1.0);
        self.last_vel = b.vel;
    }

    /// Torque at the driven axle from the engine through the gearbox, which shifts itself.
    fn drive_torque(&mut self, h: f64) -> f64 {
        let reverse = self.throttle < 0.0;
        if reverse != (self.gear < 0) {
            self.gear = if reverse { -1 } else { 1 };
        }
        let ratio = |gear: i32| {
            FINAL_DRIVE
                * if gear < 0 {
                    REVERSE
                } else {
                    GEARS[gear as usize - 1]
                }
        };
        let wheel = (self.wheels[0].spin + self.wheels[1].spin).abs() / 2.0; // the driven pair
        let to_rpm = 60.0 / std::f64::consts::TAU;
        self.rpm = (wheel * ratio(self.gear) * to_rpm).max(IDLE_RPM);
        self.shifting = (self.shifting - h).max(0.0);
        if self.gear > 0 && self.shifting == 0.0 {
            // Shift points follow the throttle, as in any automatic: gently it changes up at
            // low revs, flat out it holds each gear to near the limiter; off the throttle it
            // changes down only as the revs near idle.
            let t = self.throttle.clamp(0.0, 1.0);
            let up_at = SHIFT_UP_RPM.0 + (SHIFT_UP_RPM.1 - SHIFT_UP_RPM.0) * t;
            let down_at = SHIFT_DOWN_RPM.0 + (SHIFT_DOWN_RPM.1 - SHIFT_DOWN_RPM.0) * t;
            // Judged by road speed (the undriven rear wheels), not the driven pair, so wheelspin
            // at a launch is not taken for revs.
            let road = (self.wheels[2].spin + self.wheels[3].spin).abs() / 2.0;
            let revs = road * ratio(self.gear) * to_rpm;
            let up = revs > up_at && (self.gear as usize) < GEARS.len();
            let down = revs < down_at && self.gear > 1;
            if up || down {
                self.gear += if up { 1 } else { -1 };
                self.shifting = SHIFT_TIME;
            }
        }
        if self.shifting > 0.0 {
            return 0.0;
        }
        let torque = engine_torque(self.rpm);
        self.throttle * torque * ratio(self.gear) * DRIVETRAIN
    }

    /// Autopilot: pure-pursuit steering toward a point a little way ahead on the circle,
    /// and a speed that alternates between 15 and 8 m/s every 8 s to show acceleration and braking.
    fn drive_round(&mut self, b: &Body, radius: f64) {
        let ahead = 6.0 + 0.6 * b.vel.length();
        let angle = b.pos.z.atan2(b.pos.x) - ahead / radius; // clockwise seen from above
        let target = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
        let local = b.rot.inverse() * (target - b.pos);
        let curvature = -2.0 * local.z / local.length_squared();
        // Steering for the curve, plus the extra an understeering car needs as speed rises.
        let speed2 = b.vel.length_squared();
        self.steer = ((WHEELBASE * curvature).atan() * (1.0 + UNDERSTEER * speed2))
            .clamp(-MAX_STEER, MAX_STEER);
        let mut wanted: f64 = if (self.time / 8.0) as u64 % 2 == 0 {
            15.0
        } else {
            8.0
        };
        // Yield to a person ahead who is on the road (7 m wide) or will be within 3 s, as a
        // driver watches someone stepping off the kerb. From speed v a car braking at a stops in
        // v²/2a, so the fastest it may go with d metres left is √(2·a·d). Aim to stop with its
        // centre 6 m short, the nose about 4 m from them.
        self.yielding = false;
        if let Some((p, v)) = self.pedestrian {
            let on_road = |at: Vec3| (at.x.hypot(at.z) - radius).abs() < 3.5;
            let in_the_way = (0..=3).any(|s| on_road(p + v * s as f64));
            let gap = (b.pos.z.atan2(b.pos.x) - p.z.atan2(p.x)).rem_euclid(std::f64::consts::TAU)
                * radius;
            if in_the_way && gap < 50.0 {
                self.yielding = true;
                wanted = wanted.min((2.0 * YIELD_DECELERATION * (gap - 6.0).max(0.0)).sqrt());
            }
        }
        let forward = b.vel.dot(b.rot * Vec3::X);
        if self.yielding {
            wanted = wanted.min(forward); // never speed up toward someone in the way
        }
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

/// Engine torque at `rpm`, from `TORQUE_CURVE`; none past the rev limiter.
fn engine_torque(rpm: f64) -> f64 {
    let c = TORQUE_CURVE;
    if rpm > c[c.len() - 1].0 {
        return 0.0;
    }
    let i = c.iter().position(|&(r, _)| r >= rpm).unwrap_or(0).max(1);
    let ((r0, t0), (r1, t1)) = (c[i - 1], c[i]);
    t0 + (t1 - t0) * ((rpm - r0) / (r1 - r0)).clamp(0.0, 1.0)
}
