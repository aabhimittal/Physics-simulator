//! Articulated robots in joint coordinates, the MuJoCo way: the state is one
//! number per joint (`q`) and its rate (`qd`), so joints cannot drift apart.
//!
//! Everything comes from one routine, `inverse_dynamics` (recursive
//! Newton-Euler): given a motion it returns the joint forces that produce it.
//! The mass matrix, gravity and Coriolis forces, and the effect of a force
//! applied at any point all fall out of calling it with different inputs.

use crate::collide::{inside_box, penetration};
use crate::{Body, Quat, Shape, Vec3};

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
/// Static friction's spring (N/m per corner): how hard a corner is pulled back to where it
/// landed until friction gives way.
const GROUND_HOLD: f64 = 3.0e4;
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
    /// stance leg (0 or 1), when the current step began, how many steps it has taken, when
    /// it began reaching for the ground with the swing foot (infinite until it does), and
    /// whether it is standing still (1) rather than walking (0).
    pub memory: [f64; 5],
    pub pull: Option<Pull>,
    /// Whether the links collide with the floor.
    pub ground: bool,
    /// The floor's side profile: points (x, height), x increasing, joined by straight lines
    /// (slopes, ramps; a step is two points a hair apart) and level beyond the ends. Empty is
    /// flat at height 0. It runs unchanged across z, like the walker's world.
    pub terrain: Vec<[f64; 2]>,
    /// Forces on the links from outside (contact with world bodies, see `touch`), held for
    /// the next step.
    pub external: Vec<Push>,
    /// Corrections a learned policy adds to the controller's joint targets (rad), in the
    /// controller's own order. Empty when no policy is attached.
    pub adjust: Vec<f64>,
    /// What a person or game asks of the controller: for the walker, its walking speed (m/s).
    pub command: f64,
    /// Static friction with the floor: for each link corner touching it (8 per link), the
    /// point where it is held, which it is pulled back to by a stiff spring until that pull
    /// exceeds the friction limit, when the anchor slides along (Coulomb friction).
    anchors: Vec<Option<Vec3>>,
    /// The joint angles the walker holds while standing still (empty while walking).
    held: Vec<f64>,
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
            memory: [0.0, 0.0, 0.0, f64::INFINITY, 0.0],
            pull: None,
            ground: false,
            external: Vec::new(),
            adjust: Vec::new(),
            command: 0.0,
            anchors: vec![None; 8 * n],
            terrain: Vec::new(),
            held: Vec::new(),
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
    ///
    /// Last, the ground contacts (link, point, and whether friction is holding rather than
    /// sliding), whose springs `accelerations_over` also treats implicitly.
    #[allow(clippy::type_complexity)]
    fn applied(
        &self,
        q: &[f64],
        qd: &[f64],
    ) -> (
        Vec<f64>,
        Vec<(f64, f64)>,
        Vec<Push>,
        Vec<(usize, Vec3, Vec3, bool)>,
    ) {
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
        let (mut pushes, mut contacts) = (self.external.clone(), Vec::new());
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
            // The springs are stiff, so `accelerations_over` treats them implicitly.
            let vel = self.velocities(&frames, qd);
            for (link, l) in self.links.iter().enumerate() {
                for k in 0..8 {
                    let r = frames[link].1 * (l.com + l.half * corner_sign(k));
                    let at = frames[link].0 + r;
                    let Some((depth, n)) = self.below_floor(at).filter(|_| l.mass > 0.0) else {
                        continue;
                    };
                    let v = vel[link].1 + vel[link].0.cross(r);
                    let normal = (GROUND_STIFFNESS * depth - GROUND_DAMPING * v.dot(n)).max(0.0);
                    let along = |w: Vec3| w - n * w.dot(n); // the part along the surface
                    let anchor = self.anchors[8 * link + k].unwrap_or(at);
                    let wanted = along(anchor - at) * GROUND_HOLD - along(v) * GROUND_GRIP;
                    let limit = GROUND_FRICTION * normal;
                    pushes.push(Push {
                        link,
                        at,
                        force: n * normal + wanted.clamp_length_max(limit),
                    });
                    if normal > 0.0 {
                        contacts.push((link, at, n, wanted.length() < limit));
                    }
                }
            }
        }
        (tau, stiff, pushes, contacts)
    }

    /// How fast world point `at`, fixed to link `link`, moves per unit speed of each joint:
    /// one column per joint, zero for joints not between the link and the world.
    fn point_jacobian(&self, frames: &[(Vec3, Quat)], link: usize, at: Vec3) -> Vec<Vec3> {
        let mut columns = vec![Vec3::ZERO; self.links.len()];
        let mut joint = Some(link);
        while let Some(k) = joint {
            let l = &self.links[k];
            let axis = frames[k].1 * l.axis;
            columns[k] = match l.kind {
                JointKind::Hinge => axis.cross(at - frames[k].0),
                JointKind::Slide => axis,
            };
            joint = l.parent;
        }
        columns
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
    #[allow(clippy::needless_range_loop)] // Gaussian elimination reads clearest by index
    fn accelerations_over(&self, q: &[f64], qd: &[f64], h: f64) -> Vec<f64> {
        let n = q.len();
        let (tau, stiff, pushes, contacts) = self.applied(q, qd);
        let bias = self.inverse_dynamics(q, qd, &vec![0.0; n], self.gravity, &pushes);
        let mut m = self.mass_matrix(q);
        for (i, (damping, stiffness)) in stiff.into_iter().enumerate() {
            m[i][i] += h * damping + h * h * stiffness;
        }
        // The ground springs too: a contact force F = -K·x - C·v at a point that moves
        // J·q̇ adds Jᵀ(h·C + h²·K)J. The normal spring and damper act along y; friction, while
        // it holds, is a damper in x and z. That keeps contact stable at large time steps.
        if !contacts.is_empty() {
            let frames = self.frames(q);
            for (link, at, up, holding) in contacts {
                let j = self.point_jacobian(&frames, link, at);
                let normal = h * GROUND_DAMPING + h * h * GROUND_STIFFNESS;
                let grip = if holding {
                    h * GROUND_GRIP + h * h * GROUND_HOLD
                } else {
                    0.0
                };
                // Along the surface normal, and across the surface.
                let jn: Vec<f64> = j.iter().map(|c| c.dot(up)).collect();
                let jt: Vec<Vec3> = j.iter().zip(&jn).map(|(c, d)| *c - up * *d).collect();
                for a in 0..n {
                    for b in 0..n {
                        m[a][b] += normal * jn[a] * jn[b] + grip * jt[a].dot(jt[b]);
                    }
                }
            }
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
            if self.ground {
                self.move_anchors();
            }
        }
    }

    /// Static friction's bookkeeping after a step: a corner that has just touched the floor is
    /// held where it is; one pulled harder than friction allows slides, its anchor dragged
    /// along to the friction limit; one that has left the floor lets go.
    fn move_anchors(&mut self) {
        let frames = self.frames(&self.q);
        for (link, (o, rot)) in frames.iter().enumerate() {
            let l = &self.links[link];
            for k in 0..8 {
                let at = *o + *rot * (l.com + l.half * corner_sign(k));
                let below = self.below_floor(at).filter(|_| l.mass > 0.0);
                let anchor = self.anchors[8 * link + k];
                self.anchors[8 * link + k] = match (below, anchor) {
                    (None, _) => None,
                    (Some(_), None) => Some(at),
                    (Some((depth, n)), Some(held)) => {
                        // The limit from how far the corner presses in, not the damper's share,
                        // which the contact's buzz would flick to zero, letting go each time.
                        let most = GROUND_FRICTION * GROUND_STIFFNESS * depth / GROUND_HOLD;
                        let pulled = held - at;
                        Some(at + (pulled - n * pulled.dot(n)).clamp_length_max(most))
                    }
                };
            }
        }
    }

    /// The floor's height under `x`.
    pub fn floor_height(&self, x: f64) -> f64 {
        let t = &self.terrain;
        match t.iter().position(|p| p[0] > x) {
            None => t.last().map_or(0.0, |p| p[1]),
            Some(0) => t[0][1],
            Some(i) => {
                let ([x0, y0], [x1, y1]) = (t[i - 1], t[i]);
                y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            }
        }
    }

    /// How deep point `p` is below the floor, and the way out (unit): to the nearest point of
    /// the profile, so off a slope square to it and off a step's face sideways.
    fn below_floor(&self, p: Vec3) -> Option<(f64, Vec3)> {
        let height = self.floor_height(p.x);
        if p.y >= height {
            return None;
        }
        if self.terrain.is_empty() {
            return Some((-p.y, Vec3::Y));
        }
        // The profile with level runs out past each end; only segments within a metre can
        // be nearest, as no link is that deep.
        let t = &self.terrain;
        let (first, last) = (t[0], t[t.len() - 1]);
        let ends = [
            [[first[0] - 1e3, first[1]], first],
            [last, [last[0] + 1e3, last[1]]],
        ];
        let from = t.partition_point(|q| q[0] < p.x - 1.0).saturating_sub(1);
        let to = t.partition_point(|q| q[0] <= p.x + 1.0).min(t.len() - 1);
        t[from..=to]
            .windows(2)
            .map(|w| [w[0], w[1]])
            .chain(ends)
            .map(|[a, b]| {
                let (a, b) = (Vec3::new(a[0], a[1], p.z), Vec3::new(b[0], b[1], p.z));
                let along = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
                a + (b - a) * along
            })
            .map(|q| ((q - p).length(), q))
            .min_by(|x, y| x.0.total_cmp(&y.0))
            .map(|(depth, q)| (depth, ((q - p) / depth.max(1e-9)).normalize_or(Vec3::Y)))
    }

    /// Contact between the links and the world's bodies (not the ground plane, which
    /// `ground` covers): any link corner inside a body, and any box body's corner inside a
    /// link, is pushed out by a damped spring with friction, like ground contact. Sets
    /// `external` to the forces on the links and gives the bodies the opposite impulse over
    /// a step of `h`, so momentum passes both ways.
    pub fn touch(&mut self, bodies: &mut [Body], h: f64) {
        let frames = self.frames(&self.q);
        let vel = self.velocities(&frames, &self.qd);
        let (mut on_links, mut on_bodies) = (Vec::new(), Vec::new());
        for (link, l) in self.links.iter().enumerate() {
            if l.mass == 0.0 {
                continue;
            }
            let (o, rot) = frames[link];
            let centre = o + rot * l.com;
            for (i, b) in bodies.iter().enumerate() {
                if matches!(b.shape, Shape::Plane)
                    || centre.distance(b.pos) > l.half.length() + b.radius()
                {
                    continue;
                }
                // `out` is the way the link is pushed.
                let mut press = |at: Vec3, depth: f64, out: Vec3| {
                    let v = vel[link].1 + vel[link].0.cross(at - o) - b.point_vel(at - b.pos);
                    let closing = v.dot(out);
                    let normal = (TOUCH_STIFFNESS * depth - TOUCH_DAMPING * closing).max(0.0);
                    let slide = v - out * closing;
                    let grip = (slide * -TOUCH_GRIP).clamp_length_max(TOUCH_FRICTION * normal);
                    let force = out * normal + grip;
                    on_links.push(Push { link, at, force });
                    on_bodies.push((i, at, -force * h));
                };
                for k in 0..8 {
                    let corner = centre + rot * (l.half * corner_sign(k));
                    if let Some((depth, out)) = penetration(b, corner) {
                        press(corner, depth, out);
                    }
                }
                if let Shape::Box(bh) = b.shape {
                    for k in 0..8 {
                        let corner = b.pos + b.rot * (bh * corner_sign(k));
                        if let Some((depth, out)) = inside_box(centre, rot, l.half, corner) {
                            press(corner, depth, -out);
                        }
                    }
                }
            }
        }
        self.external = on_links;
        for (i, at, impulse) in on_bodies {
            bodies[i].wake();
            let r = at - bodies[i].pos;
            bodies[i].kick(r, impulse);
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
    let goal = if ((r.time / 10.0) as u64).is_multiple_of(2) {
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
        Some(2),
        JointKind::Hinge,
        Vec3::Z,
        Vec3::ZERO,
        Vec3::Y * 0.32,
        Vec3::new(0.1, 0.32, 0.15),
        40.0,
    );
    // The sideways slide lets a push from the side (a car) move the body; nothing balances
    // it sideways, but nothing needs to: it can slide, not tip.
    let mut links = vec![
        free(None, JointKind::Slide, Vec3::X),
        free(Some(0), JointKind::Slide, Vec3::Y),
        free(Some(1), JointKind::Slide, Vec3::Z),
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
            WALKER_TORSO,
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
            Some(WALKER_TORSO),
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
    r.ground = true;
    r.controller = Some(mimic_walk);
    r.command = WALKING_SPEED;
    start_in_stride(&mut r, 0.0);
    r
}

/// What each of the walker's links is, in link order, so the page can draw it as a person.
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u32)]
pub enum Part {
    /// Not drawn: the massless joints that let the body move and turn in the plane.
    Hidden,
    Torso,
    Thigh,
    Shin,
    Foot,
    UpperArm,
    Forearm,
}

pub const WALKER_PARTS: [Part; 14] = {
    use Part::*;
    [
        Hidden, Hidden, Hidden, Torso, Thigh, Shin, Foot, Thigh, Shin, Foot, UpperArm, Forearm,
        UpperArm, Forearm,
    ]
};

/// The walker's joints: x, y and z slides (massless) carrying the torso, whose hinge is its
/// pitch; then hip, knee and ankle for each leg, and shoulder and elbow for each arm.
pub const WALKER_TORSO: usize = 3;
/// The walker's usual pace (m/s): the captured normal walk's.
pub const WALKING_SPEED: f64 = 1.36;
pub const WALKER_HIPS: [usize; 2] = [4, 7];
const SHOULDERS: [usize; 2] = [10, 12];

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

/// Stance leg this far (rad) behind the body starts the reach for the ground early.
const STANCE_BEHIND: f64 = 0.25;

/// One of a box's eight corners, as a sign per axis.
fn corner_sign(k: usize) -> Vec3 {
    Vec3::new((k & 1) as f64, (k >> 1 & 1) as f64, (k >> 2) as f64) * 2.0 - 1.0
}

/// Robot-against-body contact (`Robot::touch`): spring and damper per point pressed in, and
/// friction up to the Coulomb limit. The same as ground contact, a little less grippy.
const TOUCH_STIFFNESS: f64 = 3.0e4;
const TOUCH_DAMPING: f64 = 300.0;
const TOUCH_GRIP: f64 = 2000.0;
const TOUCH_FRICTION: f64 = 0.5;

/// The walker standing still: everything straight, elbows a little bent.
const STANDING: [f64; 11] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2];

/// Motion-capture reference at phase `phase` of a step (0 at the stance heel strike, 1 at
/// the swing foot's) when walking at `speed`, and how long a step lasts at that speed. Between
/// two captured speeds the walks are blended; below the slowest, the slow walk is blended
/// toward standing still, so speed 0 stands and anything above it steps off.
fn reference(phase: f64, speed: f64) -> ([f64; 11], f64) {
    use crate::mocap_walk::{CLIPS, Clip};
    let at = |c: &Clip| -> [f64; 11] {
        let x = phase.clamp(0.0, 1.0) * (c.step.len() - 1) as f64;
        let (i, t) = (x as usize, x.fract());
        let j = (i + 1).min(c.step.len() - 1);
        std::array::from_fn(|k| c.step[i][k] + (c.step[j][k] - c.step[i][k]) * t)
    };
    let mix =
        |a: [f64; 11], b: [f64; 11], t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
    let slowest = &CLIPS[0];
    if speed <= slowest.speed {
        let t = (speed / slowest.speed).max(0.0);
        return (mix(STANDING, at(slowest), t), slowest.step_seconds);
    }
    let i = CLIPS
        .iter()
        .rposition(|c| c.speed <= speed)
        .unwrap()
        .min(CLIPS.len() - 2);
    let (a, b) = (&CLIPS[i], &CLIPS[i + 1]);
    let t = ((speed - a.speed) / (b.speed - a.speed)).min(1.0);
    (
        mix(at(a), at(b), t),
        a.step_seconds + (b.step_seconds - a.step_seconds) * t,
    )
}

/// How far through its step the walker is, at its commanded speed's step length in time.
fn phase(r: &Robot) -> f64 {
    (r.time - r.memory[1]) / reference(0.0, pace(r)).1
}

/// The pace the walking controller works to: the command, but never below SLOWEST_WALK. To
/// stop, the walker slows to that and then stands (`mimic_walk`).
fn pace(r: &Robot) -> f64 {
    r.command.max(SLOWEST_WALK)
}

/// Slowest walking pace (m/s), and the commanded speed below which the walker stops and
/// stands once both feet are down, or above which it steps off again.
pub const SLOWEST_WALK: f64 = 0.6;
const STOP_BELOW: f64 = 0.3;

/// Imitation: SIMBICON's balance rules with the target angles taken from a recorded walk;
/// or, when asked for no speed, standing still (`stand`).
pub fn imitate(r: &mut Robot) {
    stop_or_go(r);
    if r.memory[4] == 1.0 {
        return stand(r);
    }
    let m = MIMIC;
    let [stance_side, _, steps, mut landing_start, ..] = r.memory;
    let side = stance_side as usize;
    let (stance, swing) = (WALKER_HIPS[side], WALKER_HIPS[1 - side]);
    let frames = r.frames(&r.q);
    // Standing still, no step ends; when asked to walk again, start a fresh one.
    if phase(r) > 1.5 {
        r.memory[1] = r.time;
        return imitate(r);
    }
    let phase = phase(r);
    // The trailing foot is still down early in a step (double support): only once it has
    // lifted, late in the step, can its landing end the step.
    if landing_start.is_infinite()
        && sole_height(r, &frames, swing + 2) > 0.01
        && (phase > m.reach || r.q[WALKER_TORSO] + r.q[stance] < -STANCE_BEHIND)
    {
        landing_start = r.time;
        r.memory[3] = r.time;
    }
    if r.time > landing_start && sole_height(r, &frames, swing + 2) < 0.005 {
        r.memory = [1.0 - stance_side, r.time, steps + 1.0, f64::INFINITY, 0.0];
        return imitate(r);
    }
    let mut refs = reference(phase + m.lead, pace(r)).0;
    // The policy's corrections, if any: torso, swing hip, knee, ankle, stance knee, ankle.
    for (k, a) in [0, 4, 5, 6, 2, 3].into_iter().zip(&r.adjust) {
        refs[k] += a;
    }
    let [
        lean,
        _,
        stance_knee,
        stance_ankle,
        swing_hip,
        swing_knee,
        swing_ankle,
        s0,
        e0,
        s1,
        e1,
    ] = refs;
    let (com, com_vel) = centre_of_mass(r, &frames);
    let ankle = frames[stance + 2].0;
    let (pitch, pitch_rate) = (r.q[WALKER_TORSO], r.qd[WALKER_TORSO]);
    let pd = |angle: f64, rate: f64, want: f64, kp: f64| kp * (want - angle) - 0.1 * kp * rate;
    let aim =
        swing_hip + m.distance_gain * (com.x - ankle.x) + m.speed_gain * (com_vel.x - pace(r));
    let swing_torque = pd(
        pitch + r.q[swing],
        pitch_rate + r.qd[swing],
        aim,
        m.swing_gain,
    );
    let torso = pd(pitch, pitch_rate, m.lean + lean, 600.0);
    let mut torque = vec![0.0; r.q.len()];
    torque[swing] = swing_torque;
    torque[stance] = -torso - swing_torque;
    torque[swing + 1] = pd(r.q[swing + 1], r.qd[swing + 1], swing_knee, m.swing_gain);
    let swing_ankle_gain = if sole_height(r, &frames, swing + 2) < 0.005 {
        m.ankle_stiffness
    } else {
        30.0
    };
    torque[swing + 2] = pd(
        r.q[swing + 2],
        r.qd[swing + 2],
        swing_ankle,
        swing_ankle_gain,
    );
    torque[stance + 1] = pd(
        r.q[stance + 1],
        r.qd[stance + 1],
        stance_knee,
        m.knee_stiffness,
    );
    torque[stance + 2] = pd(
        r.q[stance + 2],
        r.qd[stance + 2],
        stance_ankle,
        m.ankle_stiffness,
    );
    for (k, (shoulder, elbow)) in [(s0, e0), (s1, e1)].into_iter().enumerate() {
        let joint = SHOULDERS[if k == 0 { side } else { 1 - side }];
        torque[joint] = pd(r.q[joint], r.qd[joint], shoulder, 60.0);
        torque[joint + 1] = pd(r.q[joint + 1], r.qd[joint + 1], elbow, 30.0);
    }
    for (i, t) in torque.iter().enumerate().skip(WALKER_TORSO + 1) {
        r.ctrl[i] = t / r.links[i].gear;
    }
    r.target = Some(Vec3::new(com.x, 0.02, com.z));
}

/// The imitation controller's own settings (the joint targets come from the capture): its
/// balance feedback (as in SIMBICON), torso lean and joint stiffnesses. Found by random
/// search for a controller that stays upright; alone it only shuffles on the spot, and the
/// trained policy (`mimic_walk`) is what makes it walk.
#[derive(Clone, Copy, Debug)]
pub struct Mimic {
    pub distance_gain: f64,
    pub speed_gain: f64,
    pub lean: f64,
    /// Phase after which a landing ends the step.
    pub reach: f64,
    pub knee_stiffness: f64,
    pub ankle_stiffness: f64,
    /// How far ahead in the capture the joints aim (fraction of a step), and the swing
    /// hip and knee stiffness (N·m/rad).
    pub lead: f64,
    pub swing_gain: f64,
}

pub const MIMIC: Mimic = Mimic {
    distance_gain: 1.02,
    speed_gain: 0.51,
    lean: 0.046,
    reach: 0.51,
    knee_stiffness: 918.0,
    ankle_stiffness: 851.0,
    lead: 0.355,
    swing_gain: 489.0,
};

/// Puts the walker in the capture's pose `phase` of the way through a step, moving at its
/// speed, the lower foot on the floor.
pub fn start_in_stride(r: &mut Robot, phase: f64) {
    let (speed, seconds) = (r.command, reference(0.0, r.command).1);
    let at = |phase| {
        let [
            lean,
            stance_hip,
            stance_knee,
            stance_ankle,
            swing_hip,
            swing_knee,
            swing_ankle,
            ..,
        ] = reference(phase, speed).0;
        [
            lean,
            stance_hip - lean,
            stance_knee,
            stance_ankle,
            swing_hip - lean,
            swing_knee,
            swing_ankle,
        ]
    };
    // Joint rates from the capture too, so the joints are already moving as they should.
    let (now, soon) = (at(phase), at(phase + 0.02));
    let rate = |k: usize| (soon[k] - now[k]) / (0.02 * seconds);
    let (stance, swing) = (WALKER_HIPS[0], WALKER_HIPS[1]);
    for (k, joint) in [
        WALKER_TORSO,
        stance,
        stance + 1,
        stance + 2,
        swing,
        swing + 1,
        swing + 2,
    ]
    .into_iter()
    .enumerate()
    {
        (r.q[joint], r.qd[joint]) = (now[k], rate(k));
    }
    r.memory = [0.0, r.time - phase * seconds, 0.0, f64::INFINITY, 0.0];
    r.q[1] = 0.0;
    let frames = r.frames(&r.q);
    let low = [stance + 2, swing + 2]
        .into_iter()
        .map(|foot| sole_height(r, &frames, foot))
        .fold(f64::INFINITY, f64::min);
    r.q[1] = -low + 0.002;
    r.qd[0] = speed;
}

/// Squared difference (rad²) between the walker's pose and the capture's at the current
/// phase of its step, averaged over the leg joints and the torso.
pub fn imitation_error(r: &Robot) -> f64 {
    let m = MIMIC;
    let side = r.memory[0] as usize;
    let (stance, swing) = (WALKER_HIPS[side], WALKER_HIPS[1 - side]);
    let refs = reference(phase(r), pace(r)).0;
    let pitch = r.q[WALKER_TORSO];
    let have = [
        pitch - m.lean,
        pitch + r.q[stance],
        r.q[stance + 1],
        r.q[stance + 2],
        pitch + r.q[swing],
        r.q[swing + 1],
        r.q[swing + 2],
    ];
    have.iter()
        .zip(&refs)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / have.len() as f64
}

/// What the imitation policy sees, the same whichever leg is on the ground: phase of the
/// step (as sine and cosine, and its raw value), height, vertical and forward speed, torso
/// pitch and rate, the speed it is asked to walk at, then the stance leg's hip, knee and ankle and the swing leg's, angles and
/// then rates. Rates scaled by 0.1 to keep every input near ±1.
pub fn imitation_obs(r: &Robot) -> Vec<f64> {
    let side = r.memory[0] as usize;
    let (stance, swing) = (WALKER_HIPS[side], WALKER_HIPS[1 - side]);
    let phase = phase(r).min(1.5);
    let joints = [stance, stance + 1, stance + 2, swing, swing + 1, swing + 2];
    let tau = std::f64::consts::TAU;
    [
        (tau * phase).sin(),
        (tau * phase).cos(),
        phase,
        hip_height(r),
        r.qd[1],
        r.qd[0],
        r.q[WALKER_TORSO],
        0.1 * r.qd[WALKER_TORSO],
        pace(r),
    ]
    .into_iter()
    .chain(joints.iter().map(|&j| r.q[j]))
    .chain(joints.iter().map(|&j| 0.1 * r.qd[j]))
    .collect()
}

/// The walker's controller with the trained policy correcting it: `imitate`, steered by
/// `env::walk_policy`.
pub fn mimic_walk(r: &mut Robot) {
    use crate::env::{IMITATE_REACH, walk_policy};
    // Once down, a person stops trying to walk: the muscles let go (joint friction remains),
    // rather than the policy flailing a body it was never trained to move.
    if hip_height(r) < 0.6 || r.q[WALKER_TORSO].abs() > 1.0 {
        r.ctrl.iter_mut().for_each(|c| *c = 0.0);
        return;
    }
    if r.memory[4] == 0.0 {
        r.adjust = walk_policy(&imitation_obs(r))
            .iter()
            .map(|a| IMITATE_REACH * a.clamp(-1.0, 1.0))
            .collect();
    }
    imitate(r);
}

/// Switches the walker between walking and standing still as its command asks.
/// Asked to stop: stand at a footfall that can stop it, one where the front toe is past the
/// capture point (centre of mass + speed·√(leg / g)), the spot it must step to in order to
/// come to rest; until then, keep taking slow steps. Asked to go: step off again.
fn stop_or_go(r: &mut Robot) {
    let frames = r.frames(&r.q);
    let both_down = WALKER_HIPS
        .iter()
        .all(|&hip| sole_height(r, &frames, hip + 2) < 0.01);
    let (com, com_vel) = centre_of_mass(r, &frames);
    let toe = WALKER_HIPS
        .iter()
        .map(|&hip| frames[hip + 2].0.x + 0.16)
        .fold(f64::NEG_INFINITY, f64::max);
    let capture = com.x + com_vel.x * (0.9f64 / 9.81).sqrt();
    if r.memory[4] == 0.0 && r.command < STOP_BELOW && both_down && toe > capture {
        // Stand as this footfall left the legs, but each knee straightened and each foot
        // put down flat, as people settle out of a stride.
        r.memory[4] = 1.0;
        r.held = r.q.clone();
        for hip in WALKER_HIPS {
            r.held[hip + 1] = -0.05;
            r.held[hip + 2] = -(r.q[WALKER_TORSO] + r.held[hip] + r.held[hip + 1]);
        }
    } else if r.memory[4] == 1.0 && r.command > STOP_BELOW {
        // Step off with the rear foot: the front one is the stance foot, which puts the
        // walker in the capture's pose at a heel strike, the start of a step.
        let front = (frames[WALKER_HIPS[1] + 2].0.x > frames[WALKER_HIPS[0] + 2].0.x) as usize;
        r.memory = [front as f64, r.time, r.memory[2], f64::INFINITY, 0.0];
    }
}

/// Height of the lowest corner of `foot`'s box.
fn sole_height(r: &Robot, frames: &[(Vec3, Quat)], foot: usize) -> f64 {
    let (o, rot) = frames[foot];
    let l = &r.links[foot];
    (0..4)
        .map(|k| {
            let p = o + rot
                * (l.com + l.half * Vec3::new([-1.0, 1.0][k & 1], -1.0, [-1.0, 1.0][k >> 1]));
            p.y - r.floor_height(p.x)
        })
        .fold(f64::INFINITY, f64::min)
}

/// The walker's hip height above the floor beneath it.
pub fn hip_height(r: &Robot) -> f64 {
    r.q[1] - r.floor_height(r.q[0])
}

/// The whole body's centre of mass and its velocity.
fn centre_of_mass(r: &Robot, frames: &[(Vec3, Quat)]) -> (Vec3, Vec3) {
    let total: f64 = r.links.iter().map(|l| l.mass).sum();
    let vel = r.velocities(frames, &r.qd);
    let (p, v) = r.links.iter().zip(frames).zip(&vel).fold(
        (Vec3::ZERO, Vec3::ZERO),
        |(p, v), ((l, (o, rot)), (w, vo))| {
            let rc = *rot * l.com;
            (p + (*o + rc) * l.mass, v + (*vo + w.cross(rc)) * l.mass)
        },
    );
    (p / total, v / total)
}

/// Standing still, as people do: the ankles leaning the body to keep its centre of mass over
/// the middle of the feet (the "ankle strategy"). Feet together, the legs stand straight and
/// the torso upright; stopped mid-stride with the feet apart, every joint is held where the
/// last step left it (with both feet gripping the floor, two straight legs are a rigid frame).
fn stand(r: &mut Robot) {
    if r.held.len() != r.q.len() {
        r.held = r.q.clone();
    }
    let frames = r.frames(&r.q);
    let (com, com_vel) = centre_of_mass(r, &frames);
    // The middle of the feet, from the rear heel (6 cm behind its ankle) to the front toe
    // (16 cm ahead of its ankle).
    let ankles = WALKER_HIPS.map(|hip| frames[hip + 2].0.x);
    let middle = (ankles[0].min(ankles[1]) - 0.06 + ankles[0].max(ankles[1]) + 0.16) / 2.0;
    let lean = STAND_LEAN.0 * (com.x - middle) + STAND_LEAN.1 * com_vel.x;
    let apart = (ankles[0] - ankles[1]).abs() > 0.15;
    let (pitch, pitch_rate) = (r.q[WALKER_TORSO], r.qd[WALKER_TORSO]);
    let pd = |angle: f64, rate: f64, want: f64, kp: f64| kp * (want - angle) - 0.1 * kp * rate;
    let mut torque = vec![0.0; r.q.len()];
    for (i, t) in torque.iter_mut().enumerate().skip(WALKER_TORSO + 1) {
        *t = pd(r.q[i], r.qd[i], r.held[i], 600.0);
    }
    for &hip in &WALKER_HIPS {
        // Feet together: each thigh vertical in the world and the torso held upright through
        // both hips, so the hips help the ankles balance. Feet apart: the held A-frame.
        if !apart {
            torque[hip] = pd(pitch + r.q[hip], pitch_rate + r.qd[hip], 0.0, 400.0)
                - 0.5 * pd(pitch, pitch_rate, MIMIC.lean, 600.0);
            torque[hip + 1] = pd(r.q[hip + 1], r.qd[hip + 1], -0.05, 600.0);
        }
        torque[hip + 2] = pd(r.q[hip + 2], r.qd[hip + 2], r.held[hip + 2] + lean, 400.0);
    }
    for shoulder in SHOULDERS {
        torque[shoulder] = pd(r.q[shoulder], r.qd[shoulder], 0.0, 60.0);
        torque[shoulder + 1] = pd(r.q[shoulder + 1], r.qd[shoulder + 1], 0.2, 30.0);
    }
    for (i, t) in torque.iter().enumerate().skip(WALKER_TORSO + 1) {
        r.ctrl[i] = t / r.links[i].gear;
    }
    r.target = Some(Vec3::new(com.x, 0.02, com.z));
}

/// The ankle strategy's gains: ankle angle per metre the centre of mass is beyond the middle
/// of the feet, and per m/s it moves (negative: toe down, which tips the shins back).
const STAND_LEAN: (f64, f64) = (-4.0, -1.0);
