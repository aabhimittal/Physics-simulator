//! Articulated robots in joint coordinates, the MuJoCo way: the state is one
//! number per joint (`q`) and its rate (`qd`), so joints cannot drift apart.
//!
//! Everything comes from one routine, `inverse_dynamics` (recursive
//! Newton-Euler): given a motion it returns the joint forces that produce it.
//! The mass matrix, gravity and Coriolis forces, and the effect of a force
//! applied at any point all fall out of calling it with different inputs.

use crate::{Quat, Shape, Vec3};

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
    /// Motor force at `ctrl = 1`. 0 means the joint has no motor.
    pub gear: f64,
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
        }
    }
}

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
    pub pull: Option<Pull>,
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
            pull: None,
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
    /// at state (`q`, `qd`) under `gravity`, with `push` acting on the robot.
    pub fn inverse_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        qdd: &[f64],
        gravity: Vec3,
        push: Option<Push>,
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
        if let Some(p) = push {
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
                self.inverse_dynamics(q, &zero, &unit, Vec3::ZERO, None)
            })
            .collect();
        columns // symmetric, so columns and rows are the same
    }

    /// Joint forces from motors, joint friction and the mouse spring.
    fn applied(&self, q: &[f64], qd: &[f64]) -> (Vec<f64>, Option<Push>) {
        let tau = (0..q.len())
            .map(|i| {
                self.links[i].gear * self.ctrl[i].clamp(-1.0, 1.0) - self.links[i].damping * qd[i]
            })
            .collect();
        let push = self.pull.map(|p| {
            let (o, rot) = self.frames(q)[p.link];
            let at = o + rot * p.local;
            let total: f64 = self.links.iter().map(|l| l.mass).sum();
            Push {
                link: p.link,
                at,
                force: (p.target - at) * (60.0 * total),
            }
        });
        (tau, push)
    }

    /// Forward dynamics: solves `M qdd = applied - bias` for the accelerations.
    pub fn accelerations(&self, q: &[f64], qd: &[f64]) -> Vec<f64> {
        let n = q.len();
        let (tau, push) = self.applied(q, qd);
        let bias = self.inverse_dynamics(q, qd, &vec![0.0; n], self.gravity, push);
        let mut m = self.mass_matrix(q);
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
            let a1 = self.accelerations(&q, &v);
            let (q2, v2) = (add(&q, &v, h / 2.0), add(&v, &a1, h / 2.0));
            let a2 = self.accelerations(&q2, &v2);
            let (q3, v3) = (add(&q, &v2, h / 2.0), add(&v, &a2, h / 2.0));
            let a3 = self.accelerations(&q3, &v3);
            let (q4, v4) = (add(&q, &v3, h), add(&v, &a3, h));
            let a4 = self.accelerations(&q4, &v4);
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
    let hold = r.inverse_dynamics(&r.q, &zero, &zero, r.gravity, None); // torques that cancel gravity
    let pull = r.inverse_dynamics(&r.q, &zero, &zero, Vec3::ZERO, Some(spring)); // minus the spring's joint torques
    for i in 0..n {
        r.ctrl[i] = (hold[i] - pull[i] - 6.0 * r.qd[i]) / r.links[i].gear;
    }
}

/// Demo robots, in the order of the web page's robot scenes.
pub fn demo(id: u32) -> Robot {
    match id {
        0 => cart_pole(),
        1 => arm(),
        _ => double_pendulum(),
    }
}

/// Boxes to draw: world position, orientation and half extents, one per link.
pub fn boxes(r: &Robot) -> Vec<(Vec3, Quat, Shape)> {
    let frames = r.frames(&r.q);
    r.links
        .iter()
        .zip(frames)
        .map(|(l, (o, rot))| (o + rot * l.com, rot, Shape::Box(l.half)))
        .collect()
}
