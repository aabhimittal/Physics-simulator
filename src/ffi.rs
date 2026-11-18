//! WebAssembly interface for `web/index.html`. Plain C ABI, no bindings generator.

use crate::env::{Learner, Player, Task};
use crate::robot::{self, Pull, Robot};
use crate::vehicle::Car;
use crate::{Body, Joint, Quat, Shape, Vec3, World, scenes};
use std::cell::RefCell;

/// Scene ids from here on are robots (`robot::demo`), drawn instead of the rigid-body world.
const FIRST_ROBOT_SCENE: u32 = 6;
/// The cart-pole that learns to balance by trial and error (`env::Learner`).
const LEARNING_SCENE: u32 = 10;
/// The hopper driven by the trained policy (`env::Player`).
const TRAINED_SCENE: u32 = 11;
/// The walking person, `robot::demo(7)`. Scene 12 is the car (`scenes::CAR_SCENE`).
const WALK_SCENE: u32 = 13;

struct State {
    world: World,
    robot: Option<Robot>,
    /// Set in the learning scene, where it owns the robot being drawn and `robot` is empty.
    learner: Option<Learner>,
    /// Set in the trained-hopper scene; owns its robot in the same way.
    player: Option<Player>,
    out: Vec<f32>,
    /// Body held by the mouse. Its drag joint is always the last joint.
    grabbed: Option<usize>,
}

thread_local! {
    static STATE: RefCell<State> =
        RefCell::new(State { world: scenes::load(0), robot: None, learner: None, player: None, out: Vec::new(), grabbed: None });
}

fn with<T>(f: impl FnOnce(&mut State) -> T) -> T {
    STATE.with_borrow_mut(f)
}

#[unsafe(no_mangle)]
pub extern "C" fn load(scene: u32) {
    with(|s| {
        s.learner = (scene == LEARNING_SCENE).then(|| Learner::new(Task::Balance, 1));
        s.player = (scene == TRAINED_SCENE).then(|| Player::new(1));
        s.robot = ((FIRST_ROBOT_SCENE..LEARNING_SCENE).contains(&scene) || scene == WALK_SCENE)
            .then(|| robot::demo(scene - FIRST_ROBOT_SCENE));
        s.world = scenes::load(scene); // empty in the robot scenes
        if scene == scenes::CROSSING_SCENE {
            let (world, walker) = scenes::crossing();
            (s.world, s.robot) = (world, Some(walker));
        }
        s.grabbed = None;
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn step(dt: f64) {
    with(|s| match &mut s.robot {
        // One learner step is one control period, the same length as a frame.
        None if s.learner.is_some() => s.learner.as_mut().unwrap().tick(),
        None if s.player.is_some() => s.player.as_mut().unwrap().tick(),
        Some(r) if !s.world.cars.is_empty() => scenes::step_crossing(&mut s.world, r, dt),
        // Ground contact is stiff, so robots that touch the floor take smaller steps.
        Some(r) => r.step(dt, if r.ground { 16 } else { 8 }),
        None => s.world.step(dt),
    })
}

/// Returns a pointer to `[bodies, joints, energy]`, then per body
/// `[pos xyz, quat xyzw, shape (0 plane, 1 sphere, 2 box, 3 wheel), size xyz]`,
/// then per joint the two anchor positions. Wheels are drawn only (they are not bodies):
/// size is (radius, half width, radius) and the axle is the local y axis.
#[unsafe(no_mangle)]
pub extern "C" fn state() -> *const f32 {
    with(|s| {
        let w = &s.world;
        // What to draw: the world's bodies, or the ground plus the robot's links and its target.
        let drawn = |(pos, rot, shape): (Vec3, Quat, Shape)| match shape {
            Shape::Plane => (pos, rot, 0.0, Vec3::ZERO),
            Shape::Sphere(r) => (pos, rot, 1.0, Vec3::splat(r)),
            Shape::Box(h) => (pos, rot, 2.0, h),
        };
        let mut draw: Vec<_> = w
            .bodies
            .iter()
            .map(|b| drawn((b.pos, b.rot, b.shape)))
            .collect();
        let (radius, half_width) = Car::WHEEL_SIZE;
        for car in &w.cars {
            let wheels = car.wheel_poses(&w.bodies[car.body]);
            draw.extend(
                wheels
                    .into_iter()
                    .map(|(p, q)| (p, q, 3.0, Vec3::new(radius, half_width, radius))),
            );
        }
        let robot = s
            .learner
            .as_ref()
            .map(|l| &l.env.robot)
            .or(s.player.as_ref().map(|p| &p.env.robot))
            .or(s.robot.as_ref());
        if let Some(r) = robot {
            let mut boxes = robot::boxes(r);
            // The target goes straight after the links, ahead of any leg rods, so it keeps its colour.
            if let Some(t) = r.target {
                boxes.insert(r.links.len(), (t, Quat::IDENTITY, Shape::Sphere(0.04)));
            }
            draw.extend(boxes.into_iter().map(drawn));
        }
        let energy = robot.map_or_else(|| w.energy(), Robot::energy);
        s.out.clear();
        s.out
            .extend([draw.len() as f32, w.joints.len() as f32, energy as f32]);
        for (pos, rot, kind, size) in draw {
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
            // The robot is drawn after the world's bodies and wheels (in robot scenes, just the
            // ground), so link i is drawn body i + that many.
            let before = s.world.bodies.len() + 4 * s.world.cars.len();
            if let Some(link) = a.checked_sub(before).filter(|&l| l < r.links.len()) {
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
        if s.robot.is_some() || s.learner.is_some() || s.player.is_some() {
            return; // robots have no contact yet, so a ball would pass straight through
        }
        let mut ball = Body::new(Shape::Sphere(0.3), 4.0, Vec3::new(ox, oy, oz));
        ball.vel = Vec3::new(vx, vy, vz);
        s.world.add(ball);
    })
}

/// Learning scene: episodes finished so far.
#[unsafe(no_mangle)]
pub extern "C" fn trials() -> u32 {
    with(|s| s.learner.as_ref().map_or(0, |l| l.trials))
}

/// Learning scene: the longest the pole has stayed up, in seconds.
#[unsafe(no_mangle)]
pub extern "C" fn best_seconds() -> f64 {
    with(|s| {
        s.learner
            .as_ref()
            .map_or(0.0, |l| l.best_return.max(0.0) * crate::env::Env::DT)
    })
}

/// Trained-hopper scene: `[episode number, seconds into it, metres travelled]`.
#[unsafe(no_mangle)]
pub extern "C" fn player(field: u32) -> f64 {
    with(|s| {
        s.player.as_ref().map_or(0.0, |p| match field {
            0 => p.episode as f64,
            1 => p.seconds,
            _ => p.env.robot.q[0],
        })
    })
}

/// Car scene: keyboard driving. `throttle` and `steer` in [-1, 1]; the autopilot turns off.
#[unsafe(no_mangle)]
pub extern "C" fn drive(throttle: f64, steer: f64) {
    with(|s| {
        if let Some(car) = s.world.cars.first_mut() {
            car.drive(&s.world.bodies[car.body], throttle, steer);
        }
    })
}

/// Car scene readout: `[speed km/h, sideways acceleration in g, body roll in degrees,
/// front tyre slip angle in degrees, 1 if on autopilot, 1 if slowing for a pedestrian,
/// brake pedal 0..1]`.
#[unsafe(no_mangle)]
pub extern "C" fn car(field: u32) -> f64 {
    with(|s| {
        let Some(car) = s.world.cars.first() else {
            return 0.0;
        };
        let b = &s.world.bodies[car.body];
        let sideways = (b.rot * Vec3::Z).with_y(0.0).normalize_or_zero();
        match field {
            0 => b.vel.length() * 3.6,
            1 => car.accel.dot(sideways).abs() / 9.81,
            2 => car.roll(b).to_degrees().abs(),
            3 => {
                car.wheels[..2]
                    .iter()
                    .map(|w| w.slip.1.abs())
                    .sum::<f64>()
                    .to_degrees()
                    / 2.0
            }
            4 => car.autopilot.is_some() as u8 as f64,
            5 => car.yielding as u8 as f64,
            _ => car.brake,
        }
    })
}

/// Walking scene readout: `[average speed m/s, steps taken, steps per second]`.
#[unsafe(no_mangle)]
pub extern "C" fn walker(field: u32) -> f64 {
    with(|s| {
        s.robot.as_ref().map_or(0.0, |r| {
            let time = r.time.max(1.0);
            match field {
                0 => r.q[0] / time,
                1 => r.memory[2],
                _ => r.memory[2] / time,
            }
        })
    })
}
