//! Articulated robots in joint coordinates, the MuJoCo way: the state is one
//! number per joint (`q`) and its rate (`qd`), so joints cannot drift apart.
//!
//! Everything comes from one routine, `inverse_dynamics` (recursive
//! Newton-Euler): given a motion it returns the joint forces that produce it.
//! The mass matrix, gravity and Coriolis forces, and the effect of a force
//! applied at any point all fall out of calling it with different inputs.

use crate::{Quat, Shape, Vec3};
use std::cell::Cell;

#[derive(Clone, Copy, PartialEq)]
pub enum JointKind {
    /// Rotates about `axis`; `q` in radians.
    Hinge,
    /// Translates along `axis`; `q` in metres.
    Slide,
}

/// One rigid link and the joint that connects it to its parent.
pub struct Link {
    /// Index of the parent link, which must come earlier in the list. `None` is the world.
    pub parent: Option<usize>,
    pub kind: JointKind,
    /// Joint axis in this link's frame.
    pub axis: Vec3,
    /// Joint position in the parent's frame.
    pub pos: Vec3,
    /// Centre of mass in this link's frame.
    pub com: Vec3,
    pub mass: f64,
    /// Diagonal inertia about the centre of mass, in this link's frame.
    pub inertia: Vec3,
    /// Half extents of the box drawn (centred on `com`).
    pub half: Vec3,
    /// Joint friction torque per unit joint speed.
    pub damping: f64,
    /// Motor force at `ctrl = 1` and standstill. 0 means the joint has no motor.
    pub gear: f64,
    /// Joint speed at which the motor can no longer push: the force it can give
    /// falls in a straight line from `gear` at standstill to zero here (a DC motor's
    /// torque-speed line). Infinite means the motor is equally strong at any speed.
    pub free_speed: f64,
    /// Allowed joint travel (min, max). Past it a stiff damped spring pushes back.
    pub range: Option<(f64, f64)>,
}

impl Link {
    /// A uniform box of the given mass.
    pub fn new(
        parent: Option<usize>,
        kind: JointKind,
        axis: Vec3,
        pos: Vec3,
        com: Vec3,
        half: Vec3,
        mass: f64,
    ) -> Link {
        let sq = half * half;
        let inertia = Vec3::new(sq.y + sq.z, sq.x + sq.z, sq.x + sq.y) * (mass / 3.0);
        Link {
            parent,
            kind,
            axis,
            pos,
            com,
            mass,
            inertia,
            half,
            damping: 0.0,
            gear: 0.0,
            free_speed: f64::INFINITY,
            range: None,
        }
    }
}

/// Ground contact: spring (N/m) and damper (N·s/m) per corner, sideways grip (N·s/m), friction coefficient.
/// Joint limit spring and damper, per unit of joint travel past the limit.
const LIMIT_STIFFNESS: f64 = 2.0e4;
const LIMIT_DAMPING: f64 = 200.0;
const GROUND_STIFFNESS: f64 = 3.0e4;
const GROUND_DAMPING: f64 = 300.0;
const GROUND_GRIP: f64 = 2000.0;
const GROUND_FRICTION: f64 = 1.0;

/// A force applied at world point `at` on link `link`.
#[derive(Clone, Copy)]
pub struct Push {
    pub link: usize,
    pub at: Vec3,
    pub force: Vec3,
}

/// The user's mouse spring: pulls local point `local` of `link` toward `target`.
#[derive(Clone, Copy)]
pub struct Pull {
    pub link: usize,
    pub local: Vec3,
    pub target: Vec3,
}

pub struct Robot {
    pub links: Vec<Link>,
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    /// Motor commands, each clamped to [-1, 1] and scaled by the joint's `gear`.
    pub ctrl: Vec<f64>,
    pub gravity: Vec3,
    pub time: f64,
    /// Sets `ctrl` (and `target`) from the current state. Runs every substep, like a real control loop.
    pub controller: Option<fn(&mut Robot)>,
    /// Where the controller is trying to go, for display.
    pub target: Option<Vec3>,
    /// Whatever the controller needs to remember between calls. The walker keeps its
    /// stance leg (0 or 1), when the current step began, how many steps it has taken, and
    /// when it began reaching for the ground with the swing foot (infinite until it does).
    pub memory: [f64; 4],
    pub pull: Option<Pull>,
    /// Whether the links collide with the floor at y = 0.
    pub ground: bool,
}

impl Robot {
    pub fn new(links: Vec<Link>) -> Robot {
        let n = links.len();
        Robot {
            links,
            q: vec![0.0; n],
            qd: vec![0.0; n],
            ctrl: vec![0.0; n],
            gravity: Vec3::new(0.0, -9.81, 0.0),
            time: 0.0,
            controller: None,
            target: None,
            memory: [0.0, 0.0, 0.0, f64::INFINITY],
            pull: None,
            ground: false,
        }
    }

    /// World position and orientation of every link frame at joint positions `q`.
    pub fn frames(&self, q: &[f64]) -> Vec<(Vec3, Quat)> {
        let mut out: Vec<(Vec3, Quat)> = Vec::with_capacity(self.links.len());
        for (i, l) in self.links.iter().enumerate() {
            let (po, pr) = l.parent.map_or((Vec3::ZERO, Quat::IDENTITY), |p| out[p]);
            let anchor = po + pr * l.pos;
            out.push(match l.kind {
                JointKind::Hinge => (anchor, pr * Quat::from_axis_angle(l.axis, q[i])),
                JointKind::Slide => (anchor + pr * l.axis * q[i], pr),
            });
        }
        out
    }

    /// Recursive Newton-Euler: the joint forces needed for accelerations `qdd`
    /// at state (`q`, `qd`) under `gravity`, with `pushes` acting on the robot.
    pub fn inverse_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        qdd: &[f64],
        gravity: Vec3,
        pushes: &[Push],
    ) -> Vec<f64> {
        let n = self.links.len();
        let frames = self.frames(q);
        // Outward pass: angular velocity, angular acceleration and frame-origin
        // acceleration of each link. Gravity enters as an upward acceleration of the world.
        let mut motion = Vec::with_capacity(n);
        let mut force = vec![Vec3::ZERO; n];
        let mut moment = vec![Vec3::ZERO; n]; // about the link's frame origin
        for (i, l) in self.links.iter().enumerate() {
            let (o, rot) = frames[i];
            let (wp, alp, aop, op) =
                l.parent
                    .map_or((Vec3::ZERO, Vec3::ZERO, -gravity, Vec3::ZERO), |p| {
                        let (w, al, ao): (Vec3, Vec3, Vec3) = motion[p];
                        (w, al, ao, frames[p].0)
                    });
            let axis = rot * l.axis;
            let r = o - op;
            let carried = aop + alp.cross(r) + wp.cross(wp.cross(r)); // the parent's point under our origin
            let (w, al, ao) = match l.kind {
                JointKind::Hinge => (
                    wp + axis * qd[i],
                    alp + axis * qdd[i] + wp.cross(axis) * qd[i],
                    carried,
                ),
                JointKind::Slide => (
                    wp,
                    alp,
                    carried + wp.cross(axis) * (2.0 * qd[i]) + axis * qdd[i],
                ),
            };
            motion.push((w, al, ao));
            // Newton and Euler for this link alone.
            let rc = rot * l.com;
            let inertia = |v: Vec3| rot * (l.inertia * (rot.inverse() * v));
            let f = (ao + al.cross(rc) + w.cross(w.cross(rc))) * l.mass;
            force[i] = f;
            moment[i] = inertia(al) + w.cross(inertia(w)) + rc.cross(f);
        }
        for p in pushes {
            force[p.link] -= p.force;
            moment[p.link] -= (p.at - frames[p.link].0).cross(p.force);
        }
        // Inward pass: each joint carries its link plus everything further out.
        let mut tau = vec![0.0; n];
        for i in (0..n).rev() {
            let l = &self.links[i];
            let axis = frames[i].1 * l.axis;
            tau[i] = axis.dot(if l.kind == JointKind::Hinge {
                moment[i]
            } else {
                force[i]
            });
            if let Some(p) = l.parent {
                let (f, m) = (force[i], moment[i]);
                force[p] += f;
                moment[p] += m + (frames[i].0 - frames[p].0).cross(f);
            }
        }
        tau
    }

    /// Mass matrix: column j is the joint force that accelerates joint j alone.
    pub fn mass_matrix(&self, q: &[f64]) -> Vec<Vec<f64>> {
        let n = self.links.len();
        let zero = vec![0.0; n];
        let columns: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let mut unit = zero.clone();
                unit[j] = 1.0;
                self.inverse_dynamics(q, &zero, &unit, Vec3::ZERO, &[])
            })
            .collect();
        columns // symmetric, so columns and rows are the same
    }

    /// Angular velocity and frame-origin velocity of every link.
    fn velocities(&self, frames: &[(Vec3, Quat)], qd: &[f64]) -> Vec<(Vec3, Vec3)> {
        let mut out: Vec<(Vec3, Vec3)> = Vec::with_capacity(self.links.len());
        for (i, l) in self.links.iter().enumerate() {
            let (wp, vp, op) = l.parent.map_or((Vec3::ZERO, Vec3::ZERO, Vec3::ZERO), |p| {
                (out[p].0, out[p].1, frames[p].0)
            });
            let axis = frames[i].1 * l.axis * qd[i];
            let carried = vp + wp.cross(frames[i].0 - op);
            out.push(match l.kind {
                JointKind::Hinge => (wp + axis, carried),
                JointKind::Slide => (wp, carried + axis),
            });
        }
        out
    }

    /// Joint forces from motors and joint friction, plus the outside forces on
    /// the links: the mouse spring and ground contact.
    ///
    /// Also returns, per joint, how fast its own joint force falls with joint speed and
    /// with joint position (damping and stiffness), so `step` can treat those implicitly.
    fn applied(&self, q: &[f64], qd: &[f64]) -> (Vec<f64>, Vec<(f64, f64)>, Vec<Push>) {
        let (tau, stiff) = (0..q.len())
            .map(|i| {
                let l = &self.links[i];
                // Speed uses up the motor's voltage: moving at a fraction `back` of its free
                // speed, it can drive forward with only 1 - back of its force. It can still
                // brake at full force, and past the free speed braking is all it can do.
                let back = qd[i] / l.free_speed;
                let drive = self.ctrl[i].clamp(-1.0 - back, 1.0 - back);
                let speed_bound = drive != self.ctrl[i] && drive.abs() < 1.0;
                let mut damping = l.damping
                    + if speed_bound {
                        l.gear / l.free_speed
                    } else {
                        0.0
                    };
                let mut stiffness = 0.0;
                let mut tau = l.gear * drive.clamp(-1.0, 1.0) - l.damping * qd[i];
                if let Some((lo, hi)) = l.range {
                    let over = q[i] - q[i].clamp(lo, hi);
                    if over != 0.0 {
                        tau += -LIMIT_STIFFNESS * over - LIMIT_DAMPING * qd[i];
                        (damping, stiffness) = (damping + LIMIT_DAMPING, LIMIT_STIFFNESS);
                    }
                }
                (tau, (damping, stiffness))
            })
            .unzip();
        let frames = self.frames(q);
        let mut pushes = Vec::new();
        if let Some(p) = self.pull {
            let (o, rot) = frames[p.link];
            let at = o + rot * p.local;
            let total: f64 = self.links.iter().map(|l| l.mass).sum();
            pushes.push(Push {
                link: p.link,
                at,
                force: (p.target - at) * (60.0 * total),
            });
        }
        if self.ground {
            // Soft contact with the floor at y = 0: every box corner below it is pushed
            // up by a damped spring and held sideways by friction, up to the Coulomb limit.
            // shortcut: explicit penalty springs need small time steps; an implicit or
            // constraint-based contact solver (as in MuJoCo) is the upgrade for stiff contact.
            let vel = self.velocities(&frames, qd);
            for (link, l) in self.links.iter().enumerate() {
                for k in 0..8 {
                    let sign =
                        Vec3::new((k & 1) as f64, (k >> 1 & 1) as f64, (k >> 2) as f64) * 2.0 - 1.0;
                    let r = frames[link].1 * (l.com + l.half * sign);
                    let at = frames[link].0 + r;
                    if l.mass == 0.0 || at.y >= 0.0 {
                        continue;
                    }
                    let v = vel[link].1 + vel[link].0.cross(r);
                    let normal = (-GROUND_STIFFNESS * at.y - GROUND_DAMPING * v.y).max(0.0);
                    let slide = Vec3::new(v.x, 0.0, v.z);
                    let grip = (slide * -GROUND_GRIP).clamp_length_max(GROUND_FRICTION * normal);
                    pushes.push(Push {
                        link,
                        at,
                        force: Vec3::Y * normal + grip,
                    });
                }
            }
        }
        (tau, stiff, pushes)
    }

    /// Forward dynamics: solves `M qdd = applied - bias` for the accelerations.
    pub fn accelerations(&self, q: &[f64], qd: &[f64]) -> Vec<f64> {
        self.accelerations_over(q, qd, 0.0)
    }

    /// Accelerations for a time step `h`, with each joint's own damping and stiffness
    /// taken implicitly: solves (M + h·D + h²·K) q̈ = forces, as MuJoCo does for damping.
    /// A stiff joint damper (a limit, or a motor past its free speed) on a light link, such
    /// as a foot, otherwise overshoots every step, and the error shows up as momentum from
    /// nowhere: a fallen walker sliding faster and faster along the floor.
    fn accelerations_over(&self, q: &[f64], qd: &[f64], h: f64) -> Vec<f64> {
        let n = q.len();
        let (tau, stiff, pushes) = self.applied(q, qd);
        let bias = self.inverse_dynamics(q, qd, &vec![0.0; n], self.gravity, &pushes);
        let mut m = self.mass_matrix(q);
        for (i, (damping, stiffness)) in stiff.into_iter().enumerate() {
            m[i][i] += h * damping + h * h * stiffness;
        }
        let mut x: Vec<f64> = (0..n).map(|i| tau[i] - bias[i]).collect();
        // Gaussian elimination. M is symmetric positive definite, so no pivoting is needed.
        for i in 0..n {
            for r in i + 1..n {
                let f = m[r][i] / m[i][i];
                for c in i..n {
                    m[r][c] -= f * m[i][c];
                }
                x[r] -= f * x[i];
            }
        }
        for i in (0..n).rev() {
            for c in i + 1..n {
                x[i] -= m[i][c] * x[c];
            }
            x[i] /= m[i][i];
        }
        x
    }

    /// Advances by `dt` in `substeps` RK4 steps; motor commands are held constant within each.
    pub fn step(&mut self, dt: f64, substeps: u32) {
        let h = dt / substeps as f64;
        let add = |a: &[f64], b: &[f64], s: f64| -> Vec<f64> {
            a.iter().zip(b).map(|(a, b)| a + b * s).collect()
        };
        for _ in 0..substeps {
            if let Some(controller) = self.controller {
                controller(self);
            }
            let (q, v) = (self.q.clone(), self.qd.clone());
            let a1 = self.accelerations_over(&q, &v, h);
            let (q2, v2) = (add(&q, &v, h / 2.0), add(&v, &a1, h / 2.0));
            let a2 = self.accelerations_over(&q2, &v2, h);
            let (q3, v3) = (add(&q, &v2, h / 2.0), add(&v, &a2, h / 2.0));
            let a3 = self.accelerations_over(&q3, &v3, h);
            let (q4, v4) = (add(&q, &v3, h), add(&v, &a3, h));
            let a4 = self.accelerations_over(&q4, &v4, h);
            for i in 0..q.len() {
                self.q[i] += h / 6.0 * (v[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]);
                self.qd[i] += h / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
            }
            self.time += h;
        }
    }

    /// Kinetic plus gravitational potential energy.
    pub fn energy(&self) -> f64 {
        let m = self.mass_matrix(&self.q);
        let n = self.q.len();
        let kinetic: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * self.qd[i] * m[i][j] * self.qd[j])
            .sum();
        let frames = self.frames(&self.q);
        let potential: f64 = self
            .links
            .iter()
            .zip(&frames)
            .map(|(l, (o, rot))| -l.mass * self.gravity.dot(*o + *rot * l.com))
            .sum();
        kinetic + potential
    }
}

// ---- Demo robots and their controllers ----

/// Cart on a rail (slide along x, motor) carrying a free-swinging pole.
/// Pole angle 0 is upright; it starts hanging down.
pub fn cart_pole() -> Robot {
    let mut cart = Link::new(
        None,
        JointKind::Slide,
        Vec3::X,
        Vec3::Y * 1.2,
        Vec3::ZERO,
        Vec3::new(0.25, 0.1, 0.15),
        1.0,
    );
    cart.gear = 25.0;
    let pole = Link::new(
        Some(0),
        JointKind::Hinge,
        Vec3::Z,
        Vec3::ZERO,
        Vec3::Y * 0.5,
        Vec3::new(0.025, 0.5, 0.025),
        0.2,
    );
    let mut r = Robot::new(vec![cart, pole]);
    r.q[1] = std::f64::consts::PI - 0.05; // just off dead centre, so the swing-up has motion to amplify
    r.controller = Some(balance_pole);
    r
}

/// Swing-up by pumping energy into the pole, then linear state feedback once it is nearly upright.
fn balance_pole(r: &mut Robot) {
    let (x, v, w) = (r.q[0], r.qd[0], r.qd[1]);
    let angle =
        (r.q[1] + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    let pole = &r.links[1];
    let (m, l) = (pole.mass, pole.com.y);
    let pivot_inertia = pole.inertia.z + m * l * l;
    // Pole energy relative to resting upright.
    let energy = 0.5 * pivot_inertia * w * w + m * 9.81 * l * (angle.cos() - 1.0);
    let force = if angle.abs() < 0.35 {
        // The cart terms look backwards on purpose: to move left it must first push
        // right, so the pole leans left and can be followed.
        -60.0 * angle - 12.0 * w + 4.0 * x + 6.0 * v
    } else {
        // Push the cart so the pole gains energy while below the target and loses it above;
        // a weak spring keeps the cart near the middle of the rail.
        (-60.0 * energy * w * angle.cos()).clamp(-12.0, 12.0) - 5.0 * x - 5.0 * v
    };
    r.ctrl[0] = force / r.links[0].gear;
}

/// Two free links, no motors: a chaotic double pendulum for checking the dynamics.
pub fn double_pendulum() -> Robot {
    let half = Vec3::new(0.03, 0.4, 0.03);
    let upper = Link::new(
        None,
        JointKind::Hinge,
        Vec3::Z,
        Vec3::Y * 2.4,
        Vec3::Y * -0.4,
        half,
        1.0,
    );
    let lower = Link::new(
        Some(0),
        JointKind::Hinge,
        Vec3::Z,
        Vec3::Y * -0.8,
        Vec3::Y * -0.4,
        half,
        1.0,
    );
    let mut r = Robot::new(vec![upper, lower]);
    r.q = vec![2.2, 0.6];
    r
}

/// Four-joint arm (base yaw, then three pitch joints) whose tip follows a moving target.
pub fn arm() -> Robot {
    let specs = [
        (Vec3::Y, 0.0, 0.1, Vec3::new(0.12, 0.1, 0.12), 3.0),
        (Vec3::Z, 0.2, 0.3, Vec3::new(0.05, 0.3, 0.05), 2.0),
        (Vec3::Z, 0.6, 0.25, Vec3::new(0.04, 0.25, 0.04), 1.5),
        (Vec3::Z, 0.5, 0.1, Vec3::new(0.03, 0.1, 0.03), 0.5),
    ];
    let links = specs
        .iter()
        .enumerate()
        .map(|(i, &(axis, up, com, half, mass))| {
            let mut l = Link::new(
                i.checked_sub(1),
                JointKind::Hinge,
                axis,
                Vec3::Y * up,
                Vec3::Y * com,
                half,
                mass,
            );
            (l.gear, l.damping) = (60.0, 0.5);
            l
        });
    let mut r = Robot::new(links.collect());
    r.q = vec![0.0, -0.6, -1.0, -0.6];
    r.controller = Some(reach);
    r
}

/// Tip of the arm's last link, in that link's frame.
pub const ARM_TIP: Vec3 = Vec3::new(0.0, 0.2, 0.0);

/// Operational-space control: a virtual spring from the tip to the target,
/// mapped to joint torques, plus gravity compensation and joint damping.
fn reach(r: &mut Robot) {
    let t = r.time * 0.7;
    let target = Vec3::new(
        0.65 + 0.15 * (2.0 * t).cos(),
        0.75 + 0.3 * (2.0 * t).sin(),
        0.55 * t.sin(),
    );
    r.target = Some(target);
    let n = r.q.len();
    let zero = vec![0.0; n];
    let (o, rot) = r.frames(&r.q)[n - 1];
    let tip = o + rot * ARM_TIP;
    let spring = Push {
        link: n - 1,
        at: tip,
        force: ((target - tip) * 1500.0).clamp_length_max(120.0),
    };
    let hold = r.inverse_dynamics(&r.q, &zero, &zero, r.gravity, &[]); // torques that cancel gravity
    let pull = r.inverse_dynamics(&r.q, &zero, &zero, Vec3::ZERO, &[spring]); // minus the spring's joint torques
    for i in 0..n {
        r.ctrl[i] = (hold[i] - pull[i] - 6.0 * r.qd[i]) / r.links[i].gear;
    }
}

/// One-legged hopper, free to move in the x-y plane. The first three joints carry
/// no mass: they are the body's x, y and pitch, which is how a moving base is
/// expressed in joint coordinates. Then a hip hinge and a sliding, sprung leg.
pub fn hopper() -> Robot {
    let free =
        |parent, kind, axis| Link::new(parent, kind, axis, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0);
    let body = Link::new(
        Some(1),
        JointKind::Hinge,
        Vec3::Z,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(0.6, 0.08, 0.1), // long, so the swinging leg barely tips it
        10.0,
    );
    let mut thigh = Link::new(
        Some(2),
        JointKind::Hinge,
        Vec3::Z,
        Vec3::ZERO,
        Vec3::Y * -0.25,
        Vec3::new(0.04, 0.25, 0.04),
        1.0,
    );
    let mut shin = Link::new(
        Some(3),
        JointKind::Slide,
        Vec3::Y,
        Vec3::Y * -0.5,
        Vec3::Y * -0.25,
        Vec3::new(0.03, 0.25, 0.03),
        0.5,
    );
    (thigh.gear, shin.gear) = (100.0, 450.0); // leg pushes with about 4x body weight
    // Peak power is gear x free speed / 4: 200 W at the hip and 560 W in the leg, about what
    // electric motors on an 11.5 kg robot can give. Without this the motors have no power limit.
    (thigh.free_speed, shin.free_speed) = (8.0, 5.0);
    shin.range = Some((-0.2, 0.3)); // a real leg can only extend so far
    thigh.range = Some((-1.2, 1.2));
    let mut r = Robot::new(vec![
        free(None, JointKind::Slide, Vec3::X),
        free(Some(0), JointKind::Slide, Vec3::Y),
        body,
        thigh,
        shin,
    ]);
    r.q[1] = 1.3;
    r.ground = true;
    r.controller = Some(hop);
    r
}

/// Leg length from hip to foot with the leg spring relaxed.
const HOPPER_LEG: f64 = 1.0;

/// Raibert's hopping controller, three independent rules:
/// - the leg is a spring, and gets an extra shove while it extends on the ground (hop height);
/// - in the air, the foot is swung to where it must land to reach the wanted speed (balance and travel);
/// - on the ground, the hip levels the body (posture).
fn hop(r: &mut Robot) {
    // Walk back and forth between two marks, 10 s each way.
    let goal = if (r.time / 10.0) as u64 % 2 == 0 {
        2.0
    } else {
        -2.0
    };
    r.target = Some(Vec3::new(goal, 0.05, 0.0));
    let (x, speed, pitch, pitch_rate) = (r.q[0], r.qd[0], r.q[2], r.qd[2]);
    let (leg_angle, leg_rate) = (pitch + r.q[3], pitch_rate + r.qd[3]);
    let (compression, compression_rate) = (r.q[4], r.qd[4]);
    let (hip, rot) = r.frames(&r.q)[3];
    let on_ground = (hip + rot * Vec3::Y * (compression - HOPPER_LEG)).y < 0.01;

    let shove = if on_ground && compression_rate < 0.0 {
        HOP_SHOVE
    } else {
        0.0
    };
    let leg_force = -HOP_SPRING * compression - 30.0 * compression_rate - shove;
    let hip_torque = if on_ground {
        150.0 * pitch + 15.0 * pitch_rate
    } else {
        // Landing with the foot at the middle of the coming stance keeps the speed;
        // further ahead slows down, further back speeds up.
        let wanted = (0.8 * (goal - x)).clamp(-0.8, 0.8);
        let ahead = speed * HOP_STANCE_TIME / 2.0 + 0.06 * (speed - wanted);
        let aim = (ahead / HOPPER_LEG).clamp(-0.5, 0.5).asin();
        150.0 * (aim - leg_angle) - 10.0 * leg_rate
    };
    r.ctrl[3] = hip_torque / r.links[3].gear;
    r.ctrl[4] = leg_force / r.links[4].gear;
}

const HOP_SPRING: f64 = 3000.0;
const HOP_SHOVE: f64 = 80.0;
/// Time the foot spends on the ground: half a bounce of the body on the leg spring.
const HOP_STANCE_TIME: f64 = 0.18;

/// A walking person, 70 kg and 1.6 m from sole to the top of the torso, free to move
/// and fall in the x-y plane (like the hopper): torso, then for each leg a thigh, shin
/// and foot (links 3-5 and 6-8), then for each arm an upper arm and forearm (9-10 and
/// 11-12), joined by motors about as strong as human muscle.
pub fn walker() -> Robot {
    let free =
        |parent, kind, axis| Link::new(parent, kind, axis, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0);
    let torso = Link::new(
        Some(1),
        JointKind::Hinge,
        Vec3::Z,
        Vec3::ZERO,
        Vec3::Y * 0.32,
        Vec3::new(0.1, 0.32, 0.15),
        40.0,
    );
    let mut links = vec![
        free(None, JointKind::Slide, Vec3::X),
        free(Some(0), JointKind::Slide, Vec3::Y),
        torso,
    ];
    for side in [1.0, -1.0] {
        let hip = links.len();
        let segment = |parent, pos: Vec3, com, half, mass, strength: f64, range| {
            let mut l = Link::new(
                Some(parent),
                JointKind::Hinge,
                Vec3::Z,
                pos,
                com,
                half,
                mass,
            );
            (l.gear, l.free_speed, l.range) = (strength, 12.0, range);
            l
        };
        // The legs sit 0.1 m to either side; they move in the same plane.
        links.push(segment(
            2,
            Vec3::Z * (0.1 * side),
            Vec3::Y * -0.22,
            Vec3::new(0.06, 0.22, 0.06),
            7.0,
            300.0,
            Some((-1.2, 1.8)),
        ));
        links.push(segment(
            hip,
            Vec3::Y * -0.44,
            Vec3::Y * -0.21,
            Vec3::new(0.045, 0.21, 0.045),
            3.3,
            300.0,
            Some((-2.4, 0.0)),
        ));
        links.push(segment(
            hip + 1,
            Vec3::Y * -0.42,
            Vec3::new(0.05, -0.04, 0.0),
            Vec3::new(0.11, 0.035, 0.045),
            1.0,
            200.0,
            None,
        ));
        // No ankle limit: the foot is so light that the stiff limit damper makes RK4 blow up
        // at this time step. Its controller keeps it in range instead.
    }
    // Arms: shoulder and elbow on each side, upper arm then forearm with the hand.
    for side in [1.0, -1.0] {
        let shoulder = links.len();
        let mut upper = Link::new(
            Some(2),
            JointKind::Hinge,
            Vec3::Z,
            Vec3::new(0.0, 0.58, 0.2 * side),
            Vec3::Y * -0.14,
            Vec3::new(0.04, 0.14, 0.04),
            2.0,
        );
        let mut fore = Link::new(
            Some(shoulder),
            JointKind::Hinge,
            Vec3::Z,
            Vec3::Y * -0.28,
            Vec3::Y * -0.17,
            Vec3::new(0.035, 0.17, 0.035),
            1.5,
        );
        (upper.gear, upper.free_speed) = (80.0, 12.0);
        (fore.gear, fore.free_speed, fore.range) = (50.0, 12.0, Some((0.0, 2.4)));
        links.push(upper);
        links.push(fore);
    }
    let mut r = Robot::new(links);
    r.q[1] = WALKER_LEG + 0.005;
    r.qd[0] = 0.4;
    r.ground = true;
    r.controller = Some(walk);
    r
}

/// Hip to sole when standing straight.
const WALKER_LEG: f64 = 0.44 + 0.42 + 0.075;

/// SIMBICON (Yin, Loken and van de Panne, 2007). A four-phase cycle: swing one leg forward
/// for a fixed time, then hold until its foot lands, then swap legs. Every leg joint is a
/// spring-damper toward a target angle. Balance comes from two rules:
/// - the swing hip aims further forward the further and faster the body is ahead of the
///   stance foot (stepping into a fall stops it, as people do);
/// - the stance hip holds the torso upright in the world, taking whatever torque that needs.
fn walk(r: &mut Robot) {
    let g = GAIT.with(Cell::get);
    let [stance_side, step_start, steps, mut landing_start] = r.memory;
    let side = stance_side as usize;
    let (stance, swing) = ([3, 6][side], [6, 3][side]);
    let frames = r.frames(&r.q);
    let sole = |foot: usize| {
        let (o, rot) = frames[foot];
        let l = &r.links[foot];
        (0..4)
            .map(|k| {
                o + rot
                    * (l.com + l.half * Vec3::new([-1.0, 1.0][k & 1], -1.0, [-1.0, 1.0][k >> 1]))
            })
            .fold(f64::INFINITY, |low, p| low.min(p.y))
    };
    // Phase: swinging, then reaching for the ground until the swing foot lands. The reach
    // starts after `swing_time`, or sooner once the stance leg is well behind the body, as a
    // person reaches sooner when walking faster; a fixed time alone lets a fast body fall
    // past its stance foot with the other foot still in the air.
    if landing_start.is_infinite()
        && (r.time - step_start > g.swing_time || r.q[2] + r.q[stance] < -STANCE_BEHIND)
    {
        landing_start = r.time;
        r.memory[3] = r.time;
    }
    if r.time > landing_start && sole(swing + 2) < 0.005 {
        r.memory = [1.0 - stance_side, r.time, steps + 1.0, f64::INFINITY];
        return walk(r);
    }
    let total: f64 = r.links.iter().map(|l| l.mass).sum();
    let (com, com_vel) = {
        let vel = r.velocities(&frames, &r.qd);
        r.links.iter().zip(&frames).zip(&vel).fold(
            (Vec3::ZERO, Vec3::ZERO),
            |(p, v), ((l, (o, rot)), (w, vo))| {
                let rc = *rot * l.com;
                (p + (*o + rc) * l.mass, v + (*vo + w.cross(rc)) * l.mass)
            },
        )
    };
    let (com, com_vel) = (com / total, com_vel / total);
    let ankle = frames[stance + 2].0;
    // Muscles ramp rather than switch: the targets blend from one pose to the next over 0.1 s.
    let blend = smoothstep((r.time - landing_start) / 0.1);
    let target = g.swinging.mix(&g.landing, blend);
    let pitch = r.q[2];
    let pd = |angle: f64, rate: f64, want: f64, kp: f64| kp * (want - angle) - 0.1 * kp * rate;
    // Swing hip: aimed in the world frame, plus the balance feedback.
    let aim = target.swing_hip + g.distance_gain * (com.x - ankle.x) + g.speed_gain * com_vel.x;
    let swing_hip = pd(pitch + r.q[swing], r.qd[2] + r.qd[swing], aim, 300.0);
    let torso = pd(pitch, r.qd[2], g.lean, 600.0);
    let mut torque = vec![0.0; r.q.len()];
    torque[swing] = swing_hip;
    torque[stance] = -torso - swing_hip;
    torque[swing + 1] = pd(r.q[swing + 1], r.qd[swing + 1], target.swing_knee, 300.0);
    // Push-off: while the trailing foot is still on the ground the calf points its toe down
    // and drives the body forward; once the foot lifts the ankle relaxes toe-up for clearance.
    // As people do, it sets the walking speed: the push eases off when faster than wanted.
    // Without this the body speeds up until it stumbles. (Pushing harder when slow is left
    // out: from a standstill, before a stride exists, that throws the body backwards.)
    let effort = (1.0 + PUSH_GAIN * (g.speed - com_vel.x)).clamp(0.0, 1.0);
    torque[swing + 2] = if sole(swing + 2) < 0.005 {
        pd(
            r.q[swing + 2],
            r.qd[swing + 2],
            g.push_off * effort,
            g.ankle_stiffness,
        )
    } else {
        pd(r.q[swing + 2], r.qd[swing + 2], target.swing_ankle, 30.0)
    };
    torque[stance + 1] = pd(
        r.q[stance + 1],
        r.qd[stance + 1],
        target.stance_knee,
        g.knee_stiffness,
    );
    torque[stance + 2] = pd(
        r.q[stance + 2],
        r.qd[stance + 2],
        target.stance_ankle,
        g.ankle_stiffness,
    );
    // Arms swing opposite to the leg on their own side, about as far as people's do, elbows
    // a little bent and bending more as the arm swings forward.
    for (shoulder, hip) in [(9, 3), (11, 6)] {
        let want = -0.6 * (pitch + r.q[hip]);
        torque[shoulder] = pd(r.q[shoulder], r.qd[shoulder], want, 60.0);
        let elbow = 0.3 + 0.4 * want.max(0.0);
        torque[shoulder + 1] = pd(r.q[shoulder + 1], r.qd[shoulder + 1], elbow, 30.0);
    }
    for i in 3..r.q.len() {
        r.ctrl[i] = torque[i] / r.links[i].gear;
    }
    r.target = Some(Vec3::new(com.x, 0.02, 0.0)); // drawn under the centre of mass
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Target joint angles (rad) for one phase. Hip angles are of the thigh from vertical,
/// positive forward; knees bend negative; ankles positive lift the toe.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub swing_hip: f64,
    pub swing_knee: f64,
    pub swing_ankle: f64,
    pub stance_knee: f64,
    pub stance_ankle: f64,
}

impl Pose {
    fn mix(&self, other: &Pose, t: f64) -> Pose {
        let m = |a: f64, b: f64| a + (b - a) * t;
        Pose {
            swing_hip: m(self.swing_hip, other.swing_hip),
            swing_knee: m(self.swing_knee, other.swing_knee),
            swing_ankle: m(self.swing_ankle, other.swing_ankle),
            stance_knee: m(self.stance_knee, other.stance_knee),
            stance_ankle: m(self.stance_ankle, other.stance_ankle),
        }
    }
}

/// The walking controller's parameters.
#[derive(Clone, Copy, Debug)]
pub struct Gait {
    pub swing_time: f64,
    pub swinging: Pose,
    pub landing: Pose,
    /// Balance feedback: extra swing-hip angle per metre the body is ahead of the stance
    /// foot, and per m/s of its speed.
    pub distance_gain: f64,
    pub speed_gain: f64,
    /// Torso angle from vertical (rad), positive leaning back.
    pub lean: f64,
    /// Ankle angle the trailing foot pushes off toward before it lifts (rad, toe down
    /// negative), at the walking speed it aims for (m/s).
    pub push_off: f64,
    pub speed: f64,
    /// Stance knee and ankle stiffness (N·m/rad): how much the leg yields under the body.
    pub knee_stiffness: f64,
    pub ankle_stiffness: f64,
}

/// The walking gait, found by `tests::tune_walk` (an opt-in test) starting from the
/// SIMBICON paper's walk. See that test for what it scores.
pub const WALK: Gait = Gait {
    swing_time: 0.4984,
    swinging: Pose {
        swing_hip: 0.4110,
        swing_knee: -1.0969,
        swing_ankle: 0.3348,
        stance_knee: -0.0238,
        stance_ankle: 0.0,
    },
    landing: Pose {
        swing_hip: -0.0667,
        swing_knee: -0.0589,
        swing_ankle: 0.2372,
        stance_knee: -0.0562,
        stance_ankle: 0.0,
    },
    distance_gain: 0.6202,
    speed_gain: 0.2290,
    lean: 0.0108,
    push_off: -0.2950,
    speed: 1.6,
    knee_stiffness: 664.4,
    ankle_stiffness: 286.8,
};

thread_local! {
    /// The gait every walker uses; `with_gait` swaps it, for tuning.
    static GAIT: Cell<Gait> = const { Cell::new(WALK) };
}

/// Runs `f` with walkers on this thread using `gait`.
pub fn with_gait<T>(gait: Gait, f: impl FnOnce() -> T) -> T {
    let old = GAIT.with(|g| g.replace(gait));
    let out = f();
    GAIT.with(|g| g.set(old));
    out
}

/// Demo robots, in the order of the web page's robot scenes.
pub fn demo(id: u32) -> Robot {
    match id {
        0 => cart_pole(),
        1 => arm(),
        2 => double_pendulum(),
        3 => hopper(),
        _ => walker(),
    }
}

/// Boxes to draw: world position, orientation and half extents, one per link.
/// After those, a thin rod for each link that slides on another link (the hopper's
/// shin), reaching back into the parent so the leg is not drawn in two pieces when it extends.
pub fn boxes(r: &Robot) -> Vec<(Vec3, Quat, Shape)> {
    let frames = r.frames(&r.q);
    let links = r.links.iter().zip(&frames);
    let rods = links
        .clone()
        .filter(|(l, _)| l.kind == JointKind::Slide && l.parent.is_some() && l.mass > 0.0)
        .map(|(l, &(o, rot))| {
            (
                o,
                rot,
                Shape::Box(Vec3::splat(0.015) + l.axis.abs() * 0.235),
            )
        });
    links
        .map(|(l, &(o, rot))| (o + rot * l.com, rot, Shape::Box(l.half)))
        .chain(rods)
        .collect()
}

const PUSH_GAIN: f64 = 1.0;
const STANCE_BEHIND: f64 = 0.25;
