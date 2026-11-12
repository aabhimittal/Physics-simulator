//! Checks the joint-coordinate robot dynamics and the demo controllers.

use glam::{DVec3 as Vec3, dvec3};
use physics_simulator::robot::{self, ARM_TIP, JointKind, Link, Pull, Robot};
use std::f64::consts::{PI, TAU};

const DT: f64 = 1.0 / 60.0;

fn run(r: &mut Robot, seconds: f64) {
    for _ in 0..(seconds / DT).round() as usize {
        r.step(DT, 8);
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} is not within {tolerance} of {expected}"
    );
}

#[test]
fn bar_pendulum_has_the_textbook_period() {
    // A uniform bar of length 2 hinged at one end: T = 2π√(I/(m g d)).
    let half = dvec3(0.02, 1.0, 0.02);
    let bar = Link::new(
        None,
        JointKind::Hinge,
        Vec3::Z,
        Vec3::ZERO,
        -Vec3::Y,
        half,
        1.0,
    );
    let pivot_inertia = bar.inertia.z + 1.0;
    let mut r = Robot::new(vec![bar]);
    let amplitude = 0.1_f64;
    r.q[0] = amplitude;
    let (mut crossings, mut last) = (Vec::new(), r.q[0]);
    while crossings.len() < 3 {
        r.step(DT / 10.0, 1);
        if r.q[0] * last < 0.0 {
            crossings.push(r.time - DT / 10.0 * r.q[0] / (r.q[0] - last));
        }
        last = r.q[0];
    }
    let expected = TAU * (pivot_inertia / 9.81).sqrt() * (1.0 + amplitude * amplitude / 16.0);
    close(crossings[2] - crossings[0], expected, expected * 1e-4);
}

#[test]
fn mass_matrix_is_symmetric() {
    let r = robot::arm();
    let m = r.mass_matrix(&[0.3, -0.8, 1.1, 0.4]);
    for i in 0..4 {
        for j in 0..4 {
            close(m[i][j], m[j][i], 1e-12);
        }
    }
}

#[test]
fn free_motion_conserves_energy() {
    // Chaotic double pendulum, and a cart-pole with the motor off (exercises the slide joint).
    let mut cart = robot::cart_pole();
    cart.controller = None;
    cart.q[1] = 2.0;
    for mut r in [robot::double_pendulum(), cart] {
        let e0 = r.energy();
        run(&mut r, 20.0);
        close(r.energy(), e0, 1e-5 * e0.abs().max(1.0));
    }
}

#[test]
fn cart_pole_swings_up_balances_and_recovers_from_a_shove() {
    let mut r = robot::cart_pole();
    let upright = |r: &Robot| ((r.q[1] + PI).rem_euclid(TAU) - PI).abs();
    run(&mut r, 15.0);
    assert!(
        upright(&r) < 0.02,
        "pole is {} rad from upright after swing-up",
        upright(&r)
    );
    close(r.q[0], 0.0, 0.1);
    // Drag the pole tip half a metre sideways for a moment, then let go.
    r.pull = Some(Pull {
        link: 1,
        local: Vec3::Y,
        target: dvec3(r.q[0] + 0.5, 2.2, 0.0),
    });
    run(&mut r, 0.3);
    r.pull = None;
    run(&mut r, 15.0);
    assert!(
        upright(&r) < 0.02,
        "pole is {} rad from upright after the shove",
        upright(&r)
    );
    close(r.q[0], 0.0, 0.1);
}

#[test]
fn arm_tip_follows_the_moving_target() {
    let mut r = robot::arm();
    run(&mut r, 3.0);
    for _ in 0..(10.0 / DT) as usize {
        r.step(DT, 8);
        let (o, rot) = r.frames(&r.q)[3];
        let error = (o + rot * ARM_TIP).distance(r.target.unwrap());
        assert!(
            error < 0.05,
            "tip is {error} m from the target at t = {}",
            r.time
        );
    }
}
