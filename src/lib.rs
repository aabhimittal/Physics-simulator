//! A small 3D rigid-body physics engine.
//!
//! Solver: XPBD with substepping (Müller et al. 2020, "Detailed Rigid Body
//! Simulation with Extended Position Based Dynamics"). Each substep integrates
//! freely, fixes joint and contact errors directly on positions, then derives
//! velocities from how far the bodies actually moved.

use glam::{DQuat as Quat, DVec3 as Vec3};

mod collide;
#[cfg(target_arch = "wasm32")]
mod ffi;
pub mod robot;
pub mod scenes;

use collide::Contact;

/// Bodies slower than this (m/s, rad/s) for `SLEEP_AFTER` seconds fall asleep.
/// Without sleeping, solver noise makes resting piles creep a few mm per second.
const SLEEP_SPEED: (f64, f64) = (0.05, 0.3);
const SLEEP_AFTER: f64 = 0.5;

#[derive(Clone, Copy)]
pub enum Shape {
    /// Infinite ground: everything below the body's y position. Static bodies only.
    Plane,
    Sphere(f64),
    /// Half extents.
    Box(Vec3),
}

pub struct Body {
    pub pos: Vec3,
    pub rot: Quat,
    pub vel: Vec3,
    pub omega: Vec3,
    pub inv_mass: f64,
    /// Diagonal of the inverse inertia tensor in the body frame.
    pub inv_inertia: Vec3,
    pub shape: Shape,
    // Pose at the start of the substep, and orientation after free motion.
    prev_pos: Vec3,
    prev_rot: Quat,
    free_rot: Quat,
    /// Seconds this body has been nearly motionless. Past `SLEEP_AFTER` it is asleep.
    calm: f64,
}

impl Body {
    /// `density == 0` makes the body static.
    pub fn new(shape: Shape, density: f64, pos: Vec3) -> Body {
        let (m, i) = match shape {
            Shape::Plane => (0.0, Vec3::ZERO),
            Shape::Sphere(r) => {
                let m = density * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
                (m, Vec3::splat(0.4 * m * r * r))
            }
            Shape::Box(h) => {
                let m = density * 8.0 * h.x * h.y * h.z;
                let sq = h * h;
                (
                    m,
                    Vec3::new(sq.y + sq.z, sq.x + sq.z, sq.x + sq.y) * (m / 3.0),
                )
            }
        };
        let dynamic = m > 0.0;
        Body {
            pos,
            rot: Quat::IDENTITY,
            vel: Vec3::ZERO,
            omega: Vec3::ZERO,
            inv_mass: if dynamic { 1.0 / m } else { 0.0 },
            inv_inertia: if dynamic { i.recip() } else { Vec3::ZERO },
            shape,
            prev_pos: pos,
            prev_rot: Quat::IDENTITY,
            free_rot: Quat::IDENTITY,
            calm: 0.0,
        }
    }

    /// Takes part in the simulation: not static and not asleep. A sleeping body acts
    /// as static until something moving touches it.
    fn active(&self) -> bool {
        self.inv_mass > 0.0 && self.calm < SLEEP_AFTER
    }

    /// Active and moving right now. Only these wake their neighbours.
    fn moving(&self) -> bool {
        self.inv_mass > 0.0 && self.calm == 0.0
    }

    pub fn wake(&mut self) {
        self.calm = 0.0;
    }

    fn radius(&self) -> f64 {
        match self.shape {
            Shape::Plane => f64::INFINITY,
            Shape::Sphere(r) => r,
            Shape::Box(h) => h.length(),
        }
    }

    /// World-frame inverse inertia applied to `v`.
    fn inv_i(&self, v: Vec3) -> Vec3 {
        self.rot * (self.inv_inertia * (self.rot.inverse() * v))
    }

    /// Velocity of the point at offset `r` from the centre of mass.
    fn point_vel(&self, r: Vec3) -> Vec3 {
        self.vel + self.omega.cross(r)
    }

    /// Generalized inverse mass: how readily a push along `n` at offset `r` moves this body.
    fn w(&self, r: Vec3, n: Vec3) -> f64 {
        let rn = r.cross(n);
        if self.active() {
            self.inv_mass + rn.dot(self.inv_i(rn))
        } else {
            0.0
        }
    }

    /// Applies positional impulse `p` at offset `r`.
    fn push(&mut self, r: Vec3, p: Vec3) {
        if !self.active() {
            return;
        }
        self.pos += p * self.inv_mass;
        self.rot = (Quat::from_scaled_axis(self.inv_i(r.cross(p))) * self.rot).normalize();
    }

    /// Applies velocity impulse `p` at offset `r`.
    fn kick(&mut self, r: Vec3, p: Vec3) {
        if !self.active() {
            return;
        }
        self.vel += p * self.inv_mass;
        self.omega += self.inv_i(r.cross(p));
    }
}

/// Keeps local point `la` of body `a` exactly `length` away from local point `lb`
/// of body `b`: a rigid massless rod, or with `length == 0` a ball joint.
/// Body 0 is the static world at the origin, so its local points are world points.
#[derive(Clone, Copy)]
pub struct Joint {
    pub a: usize,
    pub b: usize,
    pub la: Vec3,
    pub lb: Vec3,
    pub length: f64,
    /// Inverse stiffness (m/N). 0 is rigid.
    pub compliance: f64,
    /// Rate (1/s) at which relative anchor velocity is removed.
    pub damping: f64,
}

pub struct World {
    /// `bodies[0]` is always the static ground plane.
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    pub gravity: Vec3,
    pub substeps: u32,
    pub restitution: f64,
    pub friction: f64,
    contacts: Vec<Contact>,
}

impl Default for World {
    fn default() -> Self {
        World {
            bodies: vec![Body::new(Shape::Plane, 0.0, Vec3::ZERO)],
            joints: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            substeps: 20,
            restitution: 0.2,
            friction: 0.6,
            contacts: Vec::new(),
        }
    }
}

impl World {
    pub fn add(&mut self, body: Body) -> usize {
        self.bodies.push(body);
        self.bodies.len() - 1
    }

    /// World positions of a joint's two anchors.
    pub fn anchors(&self, j: &Joint) -> (Vec3, Vec3) {
        let (a, b) = (&self.bodies[j.a], &self.bodies[j.b]);
        (a.pos + a.rot * j.la, b.pos + b.rot * j.lb)
    }

    /// Kinetic plus gravitational potential energy.
    pub fn energy(&self) -> f64 {
        let dynamic = self.bodies.iter().filter(|b| b.inv_mass > 0.0);
        dynamic
            .map(|b| {
                let wb = b.rot.inverse() * b.omega;
                0.5 * b.vel.length_squared() / b.inv_mass
                    + 0.5 * (wb * wb / b.inv_inertia).element_sum()
                    - self.gravity.dot(b.pos) / b.inv_mass
            })
            .sum()
    }

    /// Moves the anchors so the error `c`, measured along `n` from b's anchor to a's, goes to 0.
    /// Returns the impulse magnitude.
    fn solve_pos(
        &mut self,
        a: usize,
        b: usize,
        ra: Vec3,
        rb: Vec3,
        n: Vec3,
        c: f64,
        alpha: f64,
    ) -> f64 {
        let w = self.bodies[a].w(ra, n) + self.bodies[b].w(rb, n);
        if w == 0.0 {
            return 0.0;
        }
        let lambda = -c / (w + alpha);
        self.bodies[a].push(ra, n * lambda);
        self.bodies[b].push(rb, n * -lambda);
        lambda
    }

    /// Changes the velocity of a's anchor relative to b's by `dv`.
    fn solve_vel(&mut self, a: usize, b: usize, ra: Vec3, rb: Vec3, dv: Vec3) {
        let Some(n) = dv.try_normalize() else { return };
        let w = self.bodies[a].w(ra, n) + self.bodies[b].w(rb, n);
        if w > 0.0 {
            self.bodies[a].kick(ra, dv / w);
            self.bodies[b].kick(rb, -dv / w);
        }
    }

    /// A moving body wakes whatever it touches or is jointed to.
    fn wake_pair(&mut self, i: usize, j: usize) {
        let (mi, mj) = (self.bodies[i].moving(), self.bodies[j].moving());
        if mi {
            self.bodies[j].wake();
        }
        if mj {
            self.bodies[i].wake();
        }
    }

    fn rel_vel(&self, a: usize, b: usize, ra: Vec3, rb: Vec3) -> Vec3 {
        self.bodies[a].point_vel(ra) - self.bodies[b].point_vel(rb)
    }

    pub fn step(&mut self, dt: f64) {
        let h = dt / self.substeps as f64;
        for _ in 0..self.substeps {
            self.integrate(h);
            self.find_contacts();
            self.solve_joints(h);
            self.solve_contacts();
            self.derive_velocities(h);
            self.solve_velocities(h);
        }
    }

    fn integrate(&mut self, h: f64) {
        for b in &mut self.bodies {
            b.prev_pos = b.pos;
            b.prev_rot = b.rot;
            if !b.active() {
                continue;
            }
            b.vel += self.gravity * h;
            b.pos += b.vel * h;
            // Free rotation. In the body frame Euler's equations involve omega alone, so
            // integrate them there with RK4. Explicit Euler gains energy every step and
            // wrecks torque-free spinning within seconds.
            let (ii, i) = (b.inv_inertia, b.inv_inertia.recip());
            let f = |w: Vec3| ii * (i * w).cross(w);
            let w0 = b.rot.inverse() * b.omega;
            let k1 = f(w0);
            let k2 = f(w0 + k1 * (0.5 * h));
            let k3 = f(w0 + k2 * (0.5 * h));
            let k4 = f(w0 + k3 * h);
            let w1 = w0 + (k1 + (k2 + k3) * 2.0 + k4) * (h / 6.0);
            b.rot = (b.rot * Quat::from_scaled_axis((w0 + w1) * (0.5 * h))).normalize();
            b.omega = b.rot * w1;
            b.free_rot = b.rot;
        }
    }

    fn find_contacts(&mut self) {
        self.contacts.clear();
        // shortcut: O(n²) bounding-sphere broadphase, fine for a few hundred bodies; use a grid or BVH beyond that.
        for i in 0..self.bodies.len() {
            for j in i + 1..self.bodies.len() {
                let (a, b) = (&self.bodies[i], &self.bodies[j]);
                let reach = a.radius() + b.radius();
                if (a.active() || b.active()) && a.pos.distance_squared(b.pos) <= reach * reach {
                    let before = self.contacts.len();
                    collide::collide(&self.bodies, i, j, &mut self.contacts);
                    if self.contacts.len() > before {
                        self.wake_pair(i, j);
                    }
                }
            }
        }
    }

    fn solve_joints(&mut self, h: f64) {
        for k in 0..self.joints.len() {
            let Joint {
                a,
                b,
                la,
                lb,
                length,
                compliance,
                ..
            } = self.joints[k];
            self.wake_pair(a, b);
            let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
            let (ra, rb) = (ba.rot * la, bb.rot * lb);
            let d = ba.pos + ra - bb.pos - rb;
            // SHAKE: push along the direction the rod had when the substep began, not
            // the one it has now. That makes the step time-symmetric. Plain PBD (current
            // direction) shortens every arc a little and bleeds about 1% of a
            // pendulum's energy per second. A zero-length joint has no rod direction.
            let d_prev = ba.prev_pos + ba.prev_rot * la - bb.prev_pos - bb.prev_rot * lb;
            let Some(n) = (if length > 0.0 { d_prev } else { d }).try_normalize() else {
                continue;
            };
            // How far to move along n so the anchors end up exactly `length` apart.
            let along = d.dot(n);
            let s = (along * along - d.length_squared() + length * length)
                .max(0.0)
                .sqrt()
                - along;
            self.solve_pos(a, b, ra, rb, n, -s, compliance / (h * h));
        }
    }

    fn solve_contacts(&mut self) {
        // Normals, Jacobi style: every contact measures its depth against the same
        // poses, and a body touched by k contacts takes 1/k of each correction.
        // Solving them one after another instead over-lifts whichever corner comes
        // first (a contact can push but never pull back), and that makes stacks rock.
        let mut count = vec![0u32; self.bodies.len()];
        for c in &mut self.contacts {
            let (a, b) = (&self.bodies[c.a], &self.bodies[c.b]);
            let (ra, rb) = (a.rot * c.la, b.rot * c.lb);
            let depth = (b.pos + rb - a.pos - ra).dot(c.n);
            if depth > 0.0 {
                c.normal_impulse = depth / (a.w(ra, c.n) + b.w(rb, c.n));
                count[c.a] += a.active() as u32;
                count[c.b] += b.active() as u32;
            }
        }
        for c in &mut self.contacts {
            if c.normal_impulse > 0.0 {
                c.normal_impulse /= count[c.a].max(count[c.b]) as f64;
                let (ra, rb) = (self.bodies[c.a].rot * c.la, self.bodies[c.b].rot * c.lb);
                self.bodies[c.a].push(ra, c.n * c.normal_impulse);
                self.bodies[c.b].push(rb, c.n * -c.normal_impulse);
            }
        }
        // Friction: undo this substep's sideways slip of the contact points, but never
        // with more impulse than the Coulomb cone allows.
        for k in 0..self.contacts.len() {
            let Contact {
                a,
                b,
                la,
                lb,
                n,
                normal_impulse,
                ..
            } = self.contacts[k];
            if normal_impulse == 0.0 {
                continue;
            }
            let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
            let (ra, rb) = (ba.rot * la, bb.rot * lb);
            let slip = (ba.pos + ra - ba.prev_pos - ba.prev_rot * la)
                - (bb.pos + rb - bb.prev_pos - bb.prev_rot * lb);
            let tangential = slip - n * slip.dot(n);
            if let Some(dir) = tangential.try_normalize() {
                let w = ba.w(ra, dir) + bb.w(rb, dir);
                let c = tangential.length().min(self.friction * normal_impulse * w);
                self.solve_pos(a, b, ra, rb, dir, c, 0.0);
            }
        }
    }

    fn derive_velocities(&mut self, h: f64) {
        for b in &mut self.bodies {
            if b.active() {
                b.vel = (b.pos - b.prev_pos) / h;
                // Only the rotation the constraints added: re-deriving all of omega from
                // the orientations would throw away the accuracy of the free-spin integrator.
                b.omega += (b.rot * b.free_rot.inverse()).to_scaled_axis() / h;
            }
        }
    }

    fn solve_velocities(&mut self, h: f64) {
        for k in 0..self.joints.len() {
            let Joint {
                a,
                b,
                la,
                lb,
                damping,
                ..
            } = self.joints[k];
            if damping > 0.0 {
                let (ra, rb) = (self.bodies[a].rot * la, self.bodies[b].rot * lb);
                let dv = self.rel_vel(a, b, ra, rb) * -(damping * h).min(1.0);
                self.solve_vel(a, b, ra, rb, dv);
            }
        }
        // Set each penetrating contact's normal velocity: a bounce for real impacts,
        // zero otherwise. Zero also cancels the speed the position pass added while
        // pushing the bodies apart, which would otherwise act as a spurious bounce.
        let rest_speed = 2.0 * self.gravity.length() * h;
        for k in 0..self.contacts.len() {
            let Contact {
                a,
                b,
                la,
                lb,
                n,
                vn_pre,
                normal_impulse,
            } = self.contacts[k];
            if normal_impulse > 0.0 {
                let (ra, rb) = (self.bodies[a].rot * la, self.bodies[b].rot * lb);
                let target = if vn_pre < -rest_speed {
                    -self.restitution * vn_pre
                } else {
                    0.0
                };
                self.solve_vel(
                    a,
                    b,
                    ra,
                    rb,
                    n * (target - self.rel_vel(a, b, ra, rb).dot(n)),
                );
            }
        }
        for b in &mut self.bodies {
            let slow = b.vel.length() < SLEEP_SPEED.0 && b.omega.length() < SLEEP_SPEED.1;
            b.calm = if slow { b.calm + h } else { 0.0 };
            if !b.active() {
                (b.vel, b.omega) = (Vec3::ZERO, Vec3::ZERO);
            }
        }
    }
}
