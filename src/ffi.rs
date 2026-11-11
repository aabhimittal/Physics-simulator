//! WebAssembly interface for `web/index.html`. Plain C ABI, no bindings generator.

use crate::{Body, Joint, Shape, Vec3, World, scenes};
use std::cell::RefCell;

struct State {
    world: World,
    out: Vec<f32>,
    /// Body held by the mouse. Its drag joint is always the last joint.
    grabbed: Option<usize>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State { world: scenes::load(0), out: Vec::new(), grabbed: None });
}

fn with<T>(f: impl FnOnce(&mut State) -> T) -> T {
    STATE.with_borrow_mut(f)
}

#[unsafe(no_mangle)]
pub extern "C" fn load(scene: u32) {
    with(|s| {
        s.world = scenes::load(scene);
        s.grabbed = None;
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn step(dt: f64) {
    with(|s| s.world.step(dt))
}

/// Returns a pointer to `[bodies, joints, energy]`, then per body
/// `[pos xyz, quat xyzw, shape (0 plane, 1 sphere, 2 box), size xyz]`,
/// then per joint the two anchor positions.
#[unsafe(no_mangle)]
pub extern "C" fn state() -> *const f32 {
    with(|s| {
        let w = &s.world;
        s.out.clear();
        s.out.extend([
            w.bodies.len() as f32,
            w.joints.len() as f32,
            w.energy() as f32,
        ]);
        for b in &w.bodies {
            let (kind, size) = match b.shape {
                Shape::Plane => (0.0, Vec3::ZERO),
                Shape::Sphere(r) => (1.0, Vec3::splat(r)),
                Shape::Box(h) => (2.0, h),
            };
            let (p, q) = (b.pos.as_vec3(), b.rot.as_quat());
            s.out.extend([
                p.x,
                p.y,
                p.z,
                q.x,
                q.y,
                q.z,
                q.w,
                kind,
                size.x as f32,
                size.y as f32,
                size.z as f32,
            ]);
        }
        for j in &w.joints {
            let (a, b) = w.anchors(j);
            s.out.extend(a.as_vec3().to_array());
            s.out.extend(b.as_vec3().to_array());
        }
        s.out.as_ptr()
    })
}

/// Starts dragging `body` by the world point (x, y, z) on it.
#[unsafe(no_mangle)]
pub extern "C" fn grab(body: u32, x: f64, y: f64, z: f64) {
    with(|s| {
        let a = body as usize;
        let Some(b) = s.world.bodies.get(a) else {
            return;
        };
        if b.inv_mass == 0.0 || s.grabbed.is_some() {
            return;
        }
        let at = Vec3::new(x, y, z);
        // A spring to the cursor, about critically damped: stiffness m·ω², ω = 30 rad/s.
        let joint = Joint {
            a,
            b: 0,
            la: b.rot.inverse() * (at - b.pos),
            lb: at,
            length: 0.0,
            compliance: b.inv_mass / 900.0,
            damping: 60.0,
        };
        s.world.joints.push(joint);
        s.grabbed = Some(a);
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn drag(x: f64, y: f64, z: f64) {
    with(|s| {
        if let (Some(a), Some(j)) = (s.grabbed, s.world.joints.last_mut()) {
            j.lb = Vec3::new(x, y, z);
            s.world.bodies[a].wake();
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn release() {
    with(|s| {
        if s.grabbed.take().is_some() {
            s.world.joints.pop();
        }
    })
}

/// Fires a ball from (ox, oy, oz) with velocity (vx, vy, vz).
#[unsafe(no_mangle)]
pub extern "C" fn shoot(ox: f64, oy: f64, oz: f64, vx: f64, vy: f64, vz: f64) {
    with(|s| {
        let mut ball = Body::new(Shape::Sphere(0.3), 4.0, Vec3::new(ox, oy, oz));
        ball.vel = Vec3::new(vx, vy, vz);
        s.world.add(ball);
    })
}
