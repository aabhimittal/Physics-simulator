//! Demo scenes. Ids match the buttons in `web/index.html`, which checks its list against
//! `kind` (through the wasm export `scene_kind`) when it loads and reports any mismatch.

use crate::robot::{self, Robot};
use crate::vehicle::Car;
use crate::{Body, Joint, Quat, Shape, Vec3, World};

/// Scene ids. 0 to 5 are rigid-body scenes (`load`); robot scenes start here, 6 to 9
/// being `robot::demo(0..=3)`.
pub const FIRST_ROBOT_SCENE: u32 = 6;
/// The cart-pole that learns to balance by trial and error (`env::Learner`).
pub const LEARNING_SCENE: u32 = 10;
/// The hopper driven by the trained policy (`env::Player`).
pub const TRAINED_SCENE: u32 = 11;
pub const CAR_SCENE: u32 = 12;
/// The walking person, `robot::walker`.
pub const WALK_SCENE: u32 = 13;
/// The person walking across the car's road; see `crossing`.
pub const CROSSING_SCENE: u32 = 14;
pub const SCENE_COUNT: u32 = 15;

/// What a scene id runs. The web page derives the same from its own scene list.
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u32)]
pub enum Kind {
    World,
    Robot,
    Learning,
    Trained,
    Car,
    Walking,
    Crossing,
}

pub fn kind(id: u32) -> Option<Kind> {
    Some(match id {
        id if id < FIRST_ROBOT_SCENE => Kind::World,
        id if id < LEARNING_SCENE => Kind::Robot,
        LEARNING_SCENE => Kind::Learning,
        TRAINED_SCENE => Kind::Trained,
        CAR_SCENE => Kind::Car,
        WALK_SCENE => Kind::Walking,
        CROSSING_SCENE => Kind::Crossing,
        _ => return None,
    })
}

/// Hangs a sphere on a rigid massless rod of length `len` from local point `lb`
/// of `parent`, swung by `angle` radians about z. Returns the new body's index.
pub fn hang(w: &mut World, parent: usize, lb: Vec3, radius: f64, len: f64, angle: f64) -> usize {
    let pivot = w.bodies[parent].pos + w.bodies[parent].rot * lb;
    let a = w.add(Body::new(
        Shape::Sphere(radius),
        1.0,
        pivot + Vec3::new(angle.sin(), -angle.cos(), 0.0) * len,
    ));
    w.joints.push(Joint {
        a,
        b: parent,
        la: Vec3::ZERO,
        lb,
        length: len,
        compliance: 0.0,
        damping: 0.0,
    });
    a
}

pub fn load(id: u32) -> World {
    let mut w = World::default();
    match id {
        // Brick wall to knock down.
        0 => {
            for row in 0..8 {
                let n = 8 - row;
                for i in 0..n {
                    let x = (i as f64 - (n - 1) as f64 / 2.0) * 0.82;
                    w.add(Body::new(
                        Shape::Box(Vec3::new(0.4, 0.2, 0.2)),
                        1.0,
                        Vec3::new(x, 0.2 + row as f64 * 0.4, 0.0),
                    ));
                }
            }
        }
        // Dominoes on an arc; the first one is already tipping.
        1 => {
            w.restitution = 0.0;
            let radius = 6.0;
            for i in 0..30 {
                let ang = i as f64 * 0.6 / radius - 1.5;
                let pos = Vec3::new(radius * ang.cos(), 0.5, radius * ang.sin());
                let mut d = Body::new(Shape::Box(Vec3::new(0.25, 0.5, 0.06)), 1.0, pos);
                d.rot = Quat::from_rotation_y(-ang); // thin axis along the arc
                if i == 0 {
                    d.omega = d.rot * Vec3::X * 2.0;
                }
                w.add(d);
            }
        }
        // Newton's cradle. The 2 mm gaps turn one five-body impact into a clean
        // chain of two-body impacts, which is also how a real cradle works.
        2 => {
            w.restitution = 1.0;
            w.friction = 0.0;
            for i in 0..5 {
                let x = (i as f64 - 2.0) * 0.502;
                hang(
                    &mut w,
                    0,
                    Vec3::new(x, 3.0, 0.0),
                    0.25,
                    2.0,
                    if i == 0 { -0.9 } else { 0.0 },
                );
            }
        }
        // Three double pendulums released 0.001 rad apart: chaos pulls them apart.
        3 => {
            for k in 0..3 {
                let angle = 2.2 + k as f64 * 1e-3;
                let upper = hang(
                    &mut w,
                    0,
                    Vec3::new(0.0, 3.4, (k as f64 - 1.0) * 0.5),
                    0.12,
                    1.0,
                    angle,
                );
                hang(&mut w, upper, Vec3::ZERO, 0.12, 1.0, angle);
            }
        }
        // Block tower: layers of three planks, each layer turned 90° from the one below.
        5 => {
            w.restitution = 0.0;
            for layer in 0..12 {
                for i in 0..3 {
                    let side = (i as f64 - 1.0) * 0.5;
                    let y = 0.15 + layer as f64 * 0.3;
                    let turned = layer % 2 == 1;
                    let pos = if turned {
                        Vec3::new(side, y, 0.0)
                    } else {
                        Vec3::new(0.0, y, side)
                    };
                    let mut plank = Body::new(Shape::Box(Vec3::new(0.75, 0.15, 0.25)), 1.0, pos);
                    if turned {
                        plank.rot = Quat::from_rotation_y(std::f64::consts::FRAC_PI_2);
                    }
                    w.add(plank);
                }
            }
        }
        // A car driving itself round a 30 m circle, past a stack of crates to drive into.
        CAR_SCENE => {
            let mut body = Body::new(
                Shape::Box(Vec3::new(2.1, 0.35, 0.7)),
                1200.0 / (8.0 * 2.1 * 0.35 * 0.7),
                Vec3::new(0.0, Car::RIDE_HEIGHT, 30.0),
            );
            body.vel = Vec3::X * 8.0;
            let body = w.add(body);
            let mut car = Car::new(body);
            car.autopilot = Some(30.0);
            for w_ in &mut car.wheels {
                w_.spin = 8.0 / Car::WHEEL_SIZE.0;
            }
            w.cars.push(car);
            for layer in 0..3 {
                for i in 0..3 - layer {
                    let x = (i as f64 - (2 - layer) as f64 / 2.0) * 0.82;
                    w.add(Body::new(
                        Shape::Box(Vec3::splat(0.4)),
                        60.0,
                        Vec3::new(x, 0.4 + layer as f64 * 0.8, 0.0),
                    ));
                }
            }
            // A speed hump across the road, 8 cm high, which the autopilot crosses every lap,
            // and a ramp inside the circle to drive up by hand. Both are static boxes; the
            // wheels' rays find them like any other surface.
            w.add(Body::new(
                Shape::Box(Vec3::new(0.6, 0.06, 3.5)),
                0.0,
                Vec3::new(0.0, 0.02, -30.0),
            ));
            let mut ramp = Body::new(
                Shape::Box(Vec3::new(2.0, 0.3, 1.5)),
                0.0,
                Vec3::new(-12.0, 0.0, 6.0),
            );
            ramp.rot = Quat::from_rotation_z(0.15);
            w.add(ramp);
            // Logs (capsules, 40 kg) to drive into or over.
            for k in 0..3 {
                let mut log = Body::new(
                    Shape::Capsule(0.15, 1.2),
                    230.0,
                    Vec3::new(-6.0 + 1.0 * k as f64, 0.15, -8.0),
                );
                log.rot = Quat::from_rotation_x(std::f64::consts::FRAC_PI_2);
                w.add(log);
            }
        }
        // Tennis racket theorem, in zero gravity: spin about the smallest and largest
        // inertia axes is stable, spin about the middle one flips over and over.
        4 => {
            w.gravity = Vec3::ZERO;
            for k in 0..3 {
                let mut b = Body::new(
                    Shape::Box(Vec3::new(0.5, 0.2, 0.05)),
                    1.0,
                    Vec3::new((k as f64 - 1.0) * 2.5, 2.5, 0.0),
                );
                b.omega = Vec3::splat(0.02);
                b.omega[k] = 6.0;
                w.add(b);
            }
        }
        _ => {} // robot scenes: an empty world
    }
    w
}

/// Where people step out: their straight path along the x axis crosses the road (radius
/// 30 m, 7 m wide) between x = 26.5 and 33.5, 2 m after this point.
const KERB: f64 = 24.5;

/// The car scene without the crates, plus a walker about to cross the road. The two live in
/// different simulators (rigid bodies and joint coordinates) and do not collide; the car
/// sees the person through its autopilot and stops for them.
pub fn crossing() -> (World, Robot) {
    let mut w = load(CAR_SCENE);
    w.bodies.truncate(2); // ground and car
    w.substeps = 2; // per step of `step_crossing`, which takes 16 steps a frame
    (w, walker_at_kerb())
}

/// The crossing scene with the car's brakes failed, timed to go wrong: the car starts 125° back
/// round the road at 50 km/h, so it reaches the zebra crossing just as the person is in its
/// lane. The driver still sees them and lifts off, but cannot stop. (Of starts every 5°, those
/// from 105° to 140° back lead to an impact; this is near the middle of that range.)
pub fn crossing_brakes_failed() -> (World, Robot) {
    let (mut w, walker) = crossing();
    let back = 125f64.to_radians();
    let along = Vec3::new(back.sin(), 0.0, -back.cos()); // the way round, clockwise from above
    let b = &mut w.bodies[1];
    b.pos = Vec3::new(30.0 * back.cos(), b.pos.y, 30.0 * back.sin());
    b.vel = along * 14.0;
    b.rot = Quat::from_rotation_y(-along.z.atan2(along.x));
    let car = &mut w.cars[0];
    car.brakes_failed = true;
    for wheel in &mut car.wheels {
        wheel.spin = 14.0 / Car::WHEEL_SIZE.0;
    }
    (w, walker)
}

fn walker_at_kerb() -> Robot {
    let mut r = robot::walker();
    r.q[0] = KERB;
    r
}

/// Advances the crossing scene one frame. Once across, the person is replaced by a new one
/// at the kerb, so someone is always about to cross.
pub fn step_crossing(w: &mut World, walker: &mut Robot, dt: f64) {
    // The walker's slides: x along its path, z sideways.
    let at = |v: &[f64]| Vec3::new(v[0], 0.0, v[2]);
    w.cars[0].pedestrian = Some((at(&walker.q), at(&walker.qd)));
    // The two simulators advance together in short steps, with contact forces between them
    // worked out afresh each time, so the person and the car can collide.
    let steps = 16;
    for _ in 0..steps {
        walker.touch(&mut w.bodies, dt / steps as f64);
        walker.step(dt / steps as f64, 1);
        w.step(dt / steps as f64);
    }
    // A new person steps off the kerb once this one is across, or once a person knocked
    // down (with the brakes failed) has come to rest, so the aftermath shows for a moment.
    let at_rest = walker.qd[..3].iter().all(|v| v.abs() < 0.1);
    if walker.q[0] > KERB + 13.0 || (walker.q[1] < 0.6 && at_rest) {
        *walker = walker_at_kerb();
    }
}
