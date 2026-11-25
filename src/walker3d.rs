//! A walking person in three dimensions: the planar walker's body (`robot::walker`) given the
//! joints it lacks to balance sideways and turn. The root can yaw, roll and pitch; each hip
//! swings forward, out to the side and twists; each ankle pitches and rolls. 22 joints.
//!
//! It walks by SIMBICON in 3D (Yin, Loken and van de Panne, 2007): the planar walker's
//! hand-tuned gait for the forward motion, and for the sideways motion the same idea turned
//! through a right angle: the swing foot lands further out the further and faster the body
//! is moving that way, and the stance hip holds the torso level and facing forward. It is a
//! hand-written controller, not a learned one.

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

/// The 3D walker, standing at the origin facing +x and moving at 0.4 m/s.
pub fn walker3d() -> Robot {
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
    r.qd[0] = 0.4;
    r.controller = Some(walk3d);
    r
}

/// The planar walker's hand-tuned SIMBICON gait (it walked without learning before motion
/// capture replaced it): two target poses a step, swinging then reaching for the ground.
struct Pose {
    swing_hip: f64,
    swing_knee: f64,
    swing_ankle: f64,
    stance_knee: f64,
}
const SWINGING: Pose = Pose {
    swing_hip: 0.411,
    swing_knee: -1.097,
    swing_ankle: 0.335,
    stance_knee: -0.024,
};
const LANDING: Pose = Pose {
    swing_hip: -0.067,
    swing_knee: -0.059,
    swing_ankle: 0.237,
    stance_knee: -0.056,
};
const SWING_TIME: f64 = 0.498;
/// Forward balance: swing hip angle per metre the body is ahead of the stance foot, and per
/// m/s; sideways, the same for the swing hip's angle out to the side.
const FORWARD_GAINS: (f64, f64) = (0.62, 0.229);
/// (Found by grids scored on shoves survived and sideways drift from a standing start: this
/// pair survives 12 of 16 shoves and drifts 1.3 m; neighbours fall, so the band is narrow.)
const SIDEWAYS_GAINS: (f64, f64) = (0.5, 0.3);
const LEAN: f64 = 0.011;
const PUSH_OFF: f64 = -0.295;
const SPEED: f64 = 1.3;
const KNEE_STIFFNESS: f64 = 664.0;
const ANKLE_STIFFNESS: f64 = 287.0;

/// SIMBICON in three dimensions; see the module notes.
pub fn walk3d(r: &mut Robot) {
    let [stance_side, step_start, steps, mut landing_start, ..] = r.memory;
    let side = stance_side as usize;
    let (stance, swing) = (LEGS[side], LEGS[1 - side]);
    let frames = r.frames(&r.q);
    let (pitch, roll, yaw) = (r.q[TORSO], r.q[ROLL], r.q[YAW]);
    if landing_start.is_infinite()
        && (r.time - step_start > SWING_TIME || pitch + r.q[stance + THIGH] < -0.25)
    {
        landing_start = r.time;
        r.memory[3] = r.time;
    }
    if r.time > landing_start && sole(r, &frames, swing + FOOT) < 0.005 {
        r.memory[..4].copy_from_slice(&[1.0 - stance_side, r.time, steps + 1.0, f64::INFINITY]);
        return walk3d(r);
    }
    let (com, com_vel) = {
        let total: f64 = r.links.iter().map(|l| l.mass).sum();
        let vel = r.velocities_at(&frames);
        let (p, v) = r.links.iter().zip(&frames).zip(&vel).fold(
            (Vec3::ZERO, Vec3::ZERO),
            |(p, v), ((l, (o, rot)), (w, vo))| {
                let rc = *rot * l.com;
                (p + (*o + rc) * l.mass, v + (*vo + w.cross(rc)) * l.mass)
            },
        );
        (p / total, v / total)
    };
    let ankle = frames[stance + FOOT].0;
    let blend = {
        let x = ((r.time - landing_start) / 0.1).clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let mix = |a: f64, b: f64| a + (b - a) * blend;
    let target = Pose {
        swing_hip: mix(SWINGING.swing_hip, LANDING.swing_hip),
        swing_knee: mix(SWINGING.swing_knee, LANDING.swing_knee),
        swing_ankle: mix(SWINGING.swing_ankle, LANDING.swing_ankle),
        stance_knee: mix(SWINGING.stance_knee, LANDING.stance_knee),
    };
    let qd = &r.qd;
    let q = &r.q;
    let pd = |angle: f64, rate: f64, want: f64, kp: f64| kp * (want - angle) - 0.1 * kp * rate;
    let mut torque = vec![0.0; q.len()];
    // Forward: SIMBICON, as in the planar walker.
    let aim = target.swing_hip + FORWARD_GAINS.0 * (com.x - ankle.x) + FORWARD_GAINS.1 * com_vel.x;
    let swing_hip = pd(
        pitch + q[swing + THIGH],
        qd[TORSO] + qd[swing + THIGH],
        aim,
        300.0,
    );
    torque[swing + THIGH] = swing_hip;
    torque[stance + THIGH] = -pd(pitch, qd[TORSO], LEAN, 600.0) - swing_hip;
    torque[swing + KNEE] = pd(q[swing + KNEE], qd[swing + KNEE], target.swing_knee, 300.0);
    let effort = (1.0 + (SPEED - com_vel.x)).clamp(0.0, 1.0);
    torque[swing + ANKLE] = if sole(r, &frames, swing + FOOT) < 0.005 {
        pd(
            q[swing + ANKLE],
            qd[swing + ANKLE],
            PUSH_OFF * effort,
            ANKLE_STIFFNESS,
        )
    } else {
        pd(
            q[swing + ANKLE],
            qd[swing + ANKLE],
            target.swing_ankle,
            30.0,
        )
    };
    torque[stance + KNEE] = pd(
        q[stance + KNEE],
        qd[stance + KNEE],
        target.stance_knee,
        KNEE_STIFFNESS,
    );
    torque[stance + ANKLE] = pd(q[stance + ANKLE], qd[stance + ANKLE], 0.0, ANKLE_STIFFNESS);
    // Sideways: the swing leg lands out the way the body drifts (a positive hip angle swings
    // the leg toward -z), the stance hip holds the torso level.
    let drift = SIDEWAYS_GAINS.0 * (com.z - ankle.z) + SIDEWAYS_GAINS.1 * com_vel.z;
    let swing_out = pd(roll + q[swing], qd[ROLL] + qd[swing], -drift, 300.0);
    torque[swing] = swing_out;
    torque[stance] = -pd(roll, qd[ROLL], 0.0, 600.0) - swing_out;
    // Turning: the stance hip holds the torso facing +x, the swing leg points straight.
    let swing_turn = pd(
        yaw + q[swing + TWIST],
        qd[YAW] + qd[swing + TWIST],
        0.0,
        100.0,
    );
    torque[swing + TWIST] = swing_turn;
    torque[stance + TWIST] = -pd(yaw, qd[YAW], 0.0, 300.0) - swing_turn;
    // Ankle roll: each foot flat across.
    for leg in [stance, swing] {
        let flat = -(roll + q[leg]);
        let kp = if leg == stance { 150.0 } else { 30.0 };
        torque[leg + FOOT] = pd(q[leg + FOOT], qd[leg + FOOT], flat, kp);
    }
    // Arms swing opposite their own side's leg.
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
    for (i, t) in torque.iter().enumerate().skip(TORSO + 1) {
        r.ctrl[i] = t / r.links[i].gear;
    }
    r.target = Some(Vec3::new(com.x, 0.02, com.z));
}

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
