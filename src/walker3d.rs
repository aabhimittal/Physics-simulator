//! A walking person in three dimensions: the planar walker's body (`robot::walker`) given the
//! joints it lacks to balance sideways and turn. The root can yaw, roll and pitch; each hip
//! swings forward, out to the side and twists; each ankle pitches and rolls. 22 joints.
//!
//! It walks as the planar walker does, by the motion-capture controller (`robot::imitate`)
//! with a learned policy correcting it, run on a planar stand-in that copies this body's
//! forward-plane joints (`drive3d`). Sideways it balances by SIMBICON's rule (Yin, Loken and
//! van de Panne, 2007) turned through a right angle: the swing foot lands further out the
//! further and faster the body is moving that way, and the stance hip holds the torso level
//! and facing forward. The policy (`env::Task::Walk3d`), trained in 3D, corrects both.

use crate::Vec3;
use crate::robot::{JointKind, Link, Robot};

/// Root joints: the torso's yaw and roll (massless links), then the torso, whose hinge is its
/// pitch.
pub const YAW: usize = 3;
pub const ROLL: usize = 4;
pub const TORSO: usize = 5;
/// Each leg's first joint (hip, sideways), left then right. After it come the hip's twist,
/// the thigh (hip forward and back), the shin (knee), the ankle's pitch, and the foot (ankle
/// roll).
pub const LEGS: [usize; 2] = [6, 12];
/// What each link is, for drawing it as a person (`robot::Part`).
pub const PARTS: [crate::robot::Part; 22] = {
    use crate::robot::Part::*;
    [
        Hidden, Hidden, Hidden, Hidden, Hidden, Torso, Hidden, Hidden, Thigh, Shin, Hidden, Foot,
        Hidden, Hidden, Thigh, Shin, Hidden, Foot, UpperArm, Forearm, UpperArm, Forearm,
    ]
};
const ARMS: [usize; 2] = [18, 20];
/// Offsets from a leg's first joint.
const TWIST: usize = 1;
const THIGH: usize = 2;
const KNEE: usize = 3;
const ANKLE: usize = 4;
const FOOT: usize = 5;

/// The 3D walker under its trained policy, starting mid-stride at walking speed, facing +x.
pub fn walker3d() -> Robot {
    let mut r = walker3d_in_stride(crate::robot::WALKING_SPEED, 0.5);
    r.controller = Some(mimic3d);
    r
}

/// The 3D walker's body: links and motors, at rest at the origin, no controller.
fn body() -> Robot {
    let free = |parent: Option<usize>, kind, axis| {
        Link::new(parent, kind, axis, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0)
    };
    let mut links = vec![
        free(None, JointKind::Slide, Vec3::X),
        free(Some(0), JointKind::Slide, Vec3::Y),
        free(Some(1), JointKind::Slide, Vec3::Z),
        free(Some(2), JointKind::Hinge, Vec3::Y),
        free(Some(3), JointKind::Hinge, Vec3::X),
        Link::new(
            Some(ROLL),
            JointKind::Hinge,
            Vec3::Z,
            Vec3::ZERO,
            Vec3::Y * 0.32,
            Vec3::new(0.1, 0.32, 0.15),
            40.0,
        ),
    ];
    let motor = |mut l: Link, strength: f64, range: Option<(f64, f64)>| {
        (l.gear, l.free_speed, l.range) = (strength, 12.0, range);
        l
    };
    for side in [1.0, -1.0] {
        let hip = links.len();
        let joint = |parent, axis, pos| {
            Link::new(
                Some(parent),
                JointKind::Hinge,
                axis,
                pos,
                Vec3::ZERO,
                Vec3::ZERO,
                0.0,
            )
        };
        links.push(motor(
            joint(TORSO, Vec3::X, Vec3::Z * (0.1 * side)),
            300.0,
            Some((-0.5, 0.5)),
        ));
        links.push(motor(
            joint(hip, Vec3::Y, Vec3::ZERO),
            150.0,
            Some((-0.6, 0.6)),
        ));
        let segment = |parent, pos, com, half, mass| {
            Link::new(
                Some(parent),
                JointKind::Hinge,
                Vec3::Z,
                pos,
                com,
                half,
                mass,
            )
        };
        links.push(motor(
            segment(
                hip + TWIST,
                Vec3::ZERO,
                Vec3::Y * -0.22,
                Vec3::new(0.06, 0.22, 0.06),
                7.0,
            ),
            300.0,
            Some((-1.2, 1.8)),
        ));
        links.push(motor(
            segment(
                hip + THIGH,
                Vec3::Y * -0.44,
                Vec3::Y * -0.21,
                Vec3::new(0.045, 0.21, 0.045),
                3.3,
            ),
            300.0,
            Some((-2.4, 0.0)),
        ));
        links.push(motor(
            joint(hip + KNEE, Vec3::Z, Vec3::Y * -0.42),
            200.0,
            None,
        ));
        // No ankle limits: the foot is so light that a stiff limit makes RK4 blow up at this
        // step; the controller keeps it in range, as in the planar walker.
        links.push(motor(
            Link::new(
                Some(hip + ANKLE),
                JointKind::Hinge,
                Vec3::X,
                Vec3::ZERO,
                Vec3::new(0.05, -0.04, 0.0),
                Vec3::new(0.11, 0.035, 0.045),
                1.0,
            ),
            150.0,
            None,
        ));
    }
    for side in [1.0, -1.0] {
        let shoulder = links.len();
        let upper = Link::new(
            Some(TORSO),
            JointKind::Hinge,
            Vec3::Z,
            Vec3::new(0.0, 0.58, 0.2 * side),
            Vec3::Y * -0.14,
            Vec3::new(0.04, 0.14, 0.04),
            2.0,
        );
        let fore = Link::new(
            Some(shoulder),
            JointKind::Hinge,
            Vec3::Z,
            Vec3::Y * -0.28,
            Vec3::Y * -0.17,
            Vec3::new(0.035, 0.17, 0.035),
            1.5,
        );
        links.push(motor(upper, 80.0, None));
        links.push(motor(fore, 50.0, Some((0.0, 2.4))));
    }
    let mut r = Robot::new(links);
    r.ground = true;
    r.q[1] = 0.44 + 0.42 + 0.075 + 0.005;
    r
}

/// Sideways balance: the swing hip's angle out to the side per metre the body is beside the
/// stance foot, and per m/s it moves that way. (Found by grids for the hand-written 3D
/// SIMBICON this walker began with, on shoves survived and drift; the policy corrects it.)
const SIDEWAYS_GAINS: (f64, f64) = (0.5, 0.3);

/// Height of the lowest corner of `foot`'s box above the floor.
fn sole(r: &Robot, frames: &[(Vec3, crate::Quat)], foot: usize) -> f64 {
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

/// The joints the 3D walker shares with the planar walker, as (3D, planar) indices: the
/// slides, the torso's pitch, each leg's hip, knee and ankle pitch, and the arms.
const SAGITTAL: [(usize, usize); 14] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (TORSO, 3),
    (8, 4),
    (9, 5),
    (10, 6),
    (14, 7),
    (15, 8),
    (16, 9),
    (18, 10),
    (19, 11),
    (20, 12),
    (21, 13),
];

/// How far (rad) the 3D policy may move the swing foot's sideways aim and the heading the
/// stance hip holds, on top of `sideways`.
pub const SIDEWAYS_REACH: f64 = 0.3;

/// The 3D walker under the walking policy for 3D (`env::walk3d_policy`, `env::Task::Walk3d`),
/// asked once a frame as it was trained; knocked down, it goes limp.
pub fn mimic3d(r: &mut Robot) {
    let down = r.q[1] - r.floor_height(r.q[0]) < 0.6 || r.q[ROLL].cos() * r.q[TORSO].cos() < 0.5;
    if down {
        r.ctrl.iter_mut().for_each(|c| *c = 0.0);
        return;
    }
    let frame_start = ((r.time * 60.0).round() - r.time * 60.0).abs() < 1e-6;
    if frame_start || r.adjust.len() != crate::env::WALK3D_ACTIONS {
        let a = crate::env::walk3d_policy(&walk3d_obs(r));
        r.adjust = crate::env::walk3d_adjust(&a);
    }
    drive3d(r);
}

/// The 3D walker driven by the planar walker's motion-capture controller (`robot::imitate`)
/// and the corrections in `Robot::adjust` (`env::WALK3D_ACTIONS` of them: the planar policy's
/// six, then the swing foot's sideways aim and the heading, in radians). Forward motion,
/// stepping and stopping are worked out on a planar stand-in that copies this body's
/// forward-plane joints; sideways balance, turning and the feet's roll by `sideways`.
pub fn drive3d(r: &mut Robot) {
    let mut planar = r
        .shadow
        .take()
        .unwrap_or_else(|| Box::new(crate::robot::walker()));
    for (a, b) in SAGITTAL {
        (planar.q[b], planar.qd[b]) = (r.q[a], r.qd[a]);
    }
    (planar.time, planar.command) = (r.time, r.command);
    let adjust = r.adjust.clone();
    planar.adjust = adjust.iter().take(6).copied().collect();
    crate::robot::imitate(&mut planar);
    let mut torque = vec![0.0; r.q.len()];
    for (a, b) in SAGITTAL.into_iter().skip(4) {
        torque[a] = planar.ctrl[b] * planar.links[b].gear;
    }
    r.memory = planar.memory;
    let frames = r.frames(&r.q);
    let offsets = [adjust.get(6), adjust.get(7)].map(|a| a.copied().unwrap_or(0.0));
    sideways(
        r,
        &frames,
        support(r, &frames, &planar),
        offsets,
        &mut torque,
    );
    swing_arms(r, &mut torque);
    for (i, t) in torque.iter().enumerate().skip(TORSO + 1) {
        r.ctrl[i] = t / r.links[i].gear;
    }
    r.target = planar.target;
    r.shadow = Some(planar);
}

/// The leg bearing the weight: the planar controller's stance leg, unless its foot is still
/// more than a centimetre above the other. (Its steps begin as the swinging foot slows for
/// the ground, a moment before it lands; holding the body's roll and heading against a leg
/// still in the air only twists the leg.)
fn support(r: &Robot, frames: &[(Vec3, crate::Quat)], planar: &Robot) -> usize {
    let side = planar.memory[0] as usize;
    let [a, b] = [side, 1 - side].map(|s| sole(r, frames, LEGS[s] + FOOT));
    if a > b + 0.01 { 1 - side } else { side }
}

/// What the 3D walking policy sees: the planar walking policy's observation of the stand-in
/// (`robot::imitation_obs`), then the torso's roll and heading and their rates, the sideways
/// speed, how far the centre of mass is beside the stance foot, and each hip's sideways angle
/// (stance, then swing). Rates scaled by 0.1.
pub fn walk3d_obs(r: &Robot) -> Vec<f64> {
    let planar = r
        .shadow
        .as_ref()
        .expect("a 3D walker under drive3d has a stand-in");
    let frames = r.frames(&r.q);
    let side = support(r, &frames, planar);
    let (com, _) = centre_of_mass(r, &frames);
    let mut obs = crate::robot::imitation_obs(planar);
    obs.extend([
        r.q[ROLL],
        0.1 * r.qd[ROLL],
        r.q[YAW],
        0.1 * r.qd[YAW],
        r.qd[2],
        com.z - frames[LEGS[side] + FOOT].0.z,
        r.q[LEGS[side]],
        r.q[LEGS[1 - side]],
    ]);
    obs
}

/// The 3D walker started as the planar walker starts in stride (`robot::start_in_stride`):
/// at `pace` m/s, `phase` of the way through a step, under `drive3d` (no policy yet).
pub fn walker3d_in_stride(pace: f64, phase: f64) -> Robot {
    let mut r = body();
    let mut planar = crate::robot::walker();
    planar.command = pace;
    crate::robot::start_in_stride(&mut planar, phase);
    for (a, b) in SAGITTAL {
        (r.q[a], r.qd[a]) = (planar.q[b], planar.qd[b]);
    }
    (r.memory, r.command) = (planar.memory, pace);
    r.shadow = Some(Box::new(planar));
    r.controller = Some(drive3d);
    r
}

/// Centre of mass and its velocity.
fn centre_of_mass(r: &Robot, frames: &[(Vec3, crate::Quat)]) -> (Vec3, Vec3) {
    let total: f64 = r.links.iter().map(|l| l.mass).sum();
    let vel = r.velocities_at(frames);
    let (p, v) = r.links.iter().zip(frames).zip(&vel).fold(
        (Vec3::ZERO, Vec3::ZERO),
        |(p, v), ((l, (o, rot)), (w, vo))| {
            let rc = *rot * l.com;
            (p + (*o + rc) * l.mass, v + (*vo + w.cross(rc)) * l.mass)
        },
    );
    (p / total, v / total)
}

/// Sideways balance, turning and the feet's roll, with leg `side` on the ground: the swing
/// leg lands out the way the body drifts (a positive hip angle swings the leg toward -z), the
/// stance hip holds the torso level and facing +x, the swing leg points straight, and each
/// foot lies flat across. `offsets` move the swing leg's sideways aim and the heading held.
fn sideways(
    r: &Robot,
    frames: &[(Vec3, crate::Quat)],
    side: usize,
    offsets: [f64; 2],
    torque: &mut [f64],
) {
    let (stance, swing) = (LEGS[side], LEGS[1 - side]);
    let (q, qd) = (&r.q, &r.qd);
    let (roll, yaw) = (q[ROLL], q[YAW]);
    let pd = |angle: f64, rate: f64, want: f64, kp: f64| kp * (want - angle) - 0.1 * kp * rate;
    let (com, com_vel) = centre_of_mass(r, frames);
    let ankle = frames[stance + FOOT].0;
    let drift = SIDEWAYS_GAINS.0 * (com.z - ankle.z) + SIDEWAYS_GAINS.1 * com_vel.z;
    let swing_out = pd(
        roll + q[swing],
        qd[ROLL] + qd[swing],
        offsets[0] - drift,
        300.0,
    );
    torque[swing] = swing_out;
    torque[stance] = -pd(roll, qd[ROLL], 0.0, 600.0) - swing_out;
    let swing_turn = pd(
        yaw + q[swing + TWIST],
        qd[YAW] + qd[swing + TWIST],
        0.0,
        100.0,
    );
    torque[swing + TWIST] = swing_turn;
    torque[stance + TWIST] = -pd(yaw, qd[YAW], offsets[1], 300.0) - swing_turn;
    for leg in [stance, swing] {
        let flat = -(roll + q[leg]);
        let kp = if leg == stance { 150.0 } else { 30.0 };
        torque[leg + FOOT] = pd(q[leg + FOOT], qd[leg + FOOT], flat, kp);
    }
}

/// Arms swing opposite their own side's leg, as people's do, which also cancels some of the
/// legs' twist on the body.
fn swing_arms(r: &Robot, torque: &mut [f64]) {
    let (q, qd, pitch) = (&r.q, &r.qd, r.q[TORSO]);
    let pd = |angle: f64, rate: f64, want: f64, kp: f64| kp * (want - angle) - 0.1 * kp * rate;
    for (shoulder, leg) in ARMS.into_iter().zip(LEGS) {
        let want = -0.6 * (pitch + q[leg + THIGH]);
        torque[shoulder] = pd(q[shoulder], qd[shoulder], want, 60.0);
        torque[shoulder + 1] = pd(
            q[shoulder + 1],
            qd[shoulder + 1],
            0.3 + 0.4 * want.max(0.0),
            30.0,
        );
    }
}
