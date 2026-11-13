//! WebAssembly interface for `web/index.html`. Plain C ABI, no bindings generator.

use crate::robot::{self, Pull, Robot};
use crate::{Body, Joint, Quat, Shape, Vec3, World, scenes};
use std::cell::RefCell;

/// Scene ids from here on are robots (`robot::demo`), drawn instead of the rigid-body world.
const FIRST_ROBOT_SCENE: u32 = 6;

struct State {
    world: World,
    robot: Option<Robot>,
    out: Vec<f32>,
    /// Body held by the mouse. Its drag joint is always the last joint.
    grabbed: Option<usize>,
}

thread_local! {
    static STATE: RefCell<State> =
        RefCell::new(State { world: scenes::load(0), robot: None, out: Vec::new(), grabbed: None });
}

fn with<T>(f: impl FnOnce(&mut State) -> T) -> T {
    STATE.with_borrow_mut(f)
}

#[unsafe(no_mangle)]
pub extern "C" fn load(scene: u32) {
    with(|s| {
        s.robot = scene.checked_sub(FIRST_ROBOT_SCENE).map(robot::demo);
        s.world = if s.robot.is_some() {
            World::default()
        } else {
            scenes::load(scene)
        };
        s.grabbed = None;
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn step(dt: f64) {
    with(|s| match &mut s.robot {
        // Ground contact is stiff, so robots that touch the floor take smaller steps.
        Some(r) => r.step(dt, if r.ground { 16 } else { 8 }),
        None => s.world.step(dt),
    })
}

/// Returns a pointer to `[bodies, joints, energy]`, then per body
/// `[pos xyz, quat xyzw, shape (0 plane, 1 sphere, 2 box), size xyz]`,
/// then per joint the two anchor positions.
#[unsafe(no_mangle)]
pub extern "C" fn state() -> *const f32 {
    with(|s| {
        let w = &s.world;
        // What to draw: the world's bodies, or the ground plus the robot's links and its target.
        let mut draw: Vec<_> = w.bodies.iter().map(|b| (b.pos, b.rot, b.shape)).collect();
        if let Some(r) = &s.robot {
            draw.extend(robot::boxes(r));
            draw.extend(r.target.map(|t| (t, Quat::IDENTITY, Shape::Sphere(0.04))));
        }
        let energy = s.robot.as_ref().map_or_else(|| w.energy(), Robot::energy);
        s.out.clear();
        s.out
            .extend([draw.len() as f32, w.joints.len() as f32, energy as f32]);
        for (pos, rot, shape) in draw {
            let (kind, size) = match shape {
                Shape::Plane => (0.0, Vec3::ZERO),
                Shape::Sphere(r) => (1.0, Vec3::splat(r)),
                Shape::Box(h) => (2.0, h),
            };
            let (p, q, size) = (pos.as_vec3(), rot.as_quat(), size.as_vec3());
            s.out.extend([
                p.x, p.y, p.z, q.x, q.y, q.z, q.w, kind, size.x, size.y, size.z,
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

/// Starts dragging drawn body `body` by the world point (x, y, z) on it.
#[unsafe(no_mangle)]
pub extern "C" fn grab(body: u32, x: f64, y: f64, z: f64) {
    with(|s| {
        let (a, at) = (body as usize, Vec3::new(x, y, z));
        if let Some(r) = &mut s.robot {
            // Drawn body 0 is the ground, so link i is drawn body i + 1.
            if let Some(link) = a.checked_sub(1).filter(|&l| l < r.links.len()) {
                let (o, rot) = r.frames(&r.q)[link];
                r.pull = Some(Pull {
                    link,
                    local: rot.inverse() * (at - o),
                    target: at,
                });
            }
            return;
        }
        let Some(b) = s.world.bodies.get(a) else {
            return;
        };
        if b.inv_mass == 0.0 || s.grabbed.is_some() {
            return;
        }
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
        if let Some(pull) = s.robot.as_mut().and_then(|r| r.pull.as_mut()) {
            pull.target = Vec3::new(x, y, z);
        }
        if let (Some(a), Some(j)) = (s.grabbed, s.world.joints.last_mut()) {
            j.lb = Vec3::new(x, y, z);
            s.world.bodies[a].wake();
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn release() {
    with(|s| {
        if let Some(r) = &mut s.robot {
            r.pull = None;
        }
        if s.grabbed.take().is_some() {
            s.world.joints.pop();
        }
    })
}

/// Fires a ball from (ox, oy, oz) with velocity (vx, vy, vz).
#[unsafe(no_mangle)]
pub extern "C" fn shoot(ox: f64, oy: f64, oz: f64, vx: f64, vy: f64, vz: f64) {
    with(|s| {
        if s.robot.is_some() {
            return; // robots have no contact yet, so a ball would pass straight through
        }
        let mut ball = Body::new(Shape::Sphere(0.3), 4.0, Vec3::new(ox, oy, oz));
        ball.vel = Vec3::new(vx, vy, vz);
        s.world.add(ball);
    })
}
