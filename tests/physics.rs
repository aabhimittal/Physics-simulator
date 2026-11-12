//! Checks the engine against physics with known answers.

use glam::{DVec3 as Vec3, dvec3};
use physics_simulator::{Body, Shape, World, scenes};

const DT: f64 = 1.0 / 60.0;

fn run(w: &mut World, seconds: f64) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} is not within {tolerance} of {expected}"
    );
}

#[test]
fn sphere_falls_like_half_g_t_squared_then_rests() {
    let mut w = World::default();
    let s = w.add(Body::new(Shape::Sphere(0.5), 1.0, dvec3(0.0, 5.0, 0.0)));
    run(&mut w, 0.5);
    close(w.bodies[s].pos.y, 5.0 - 0.5 * 9.81 * 0.25, 0.01);
    run(&mut w, 6.0);
    close(w.bodies[s].pos.y, 0.5, 0.005);
    close(w.bodies[s].vel.length(), 0.0, 0.01);
}

#[test]
fn box_stack_stays_put() {
    let mut w = World::default();
    for i in 0..6 {
        w.add(Body::new(
            Shape::Box(Vec3::splat(0.25)),
            1.0,
            dvec3(0.0, 0.25 + 0.5 * i as f64, 0.0),
        ));
    }
    run(&mut w, 8.0);
    for (i, b) in w.bodies.iter().enumerate().skip(1) {
        close(b.pos.x, 0.0, 0.01);
        close(b.pos.z, 0.0, 0.01);
        close(b.pos.y, 0.25 + 0.5 * (i - 1) as f64, 0.01);
    }
}

#[test]
fn block_tower_stands_until_hit() {
    let mut w = scenes::load(5);
    let start: Vec<Vec3> = w.bodies.iter().map(|b| b.pos).collect();
    run(&mut w, 30.0);
    // Tall stacks are slightly soft: allow 2 cm of sink at the top (3.6 m up), 1 cm sideways.
    for (b, s) in w.bodies.iter().zip(&start) {
        let moved = b.pos - *s;
        close(moved.y, 0.0, 0.02);
        close(moved.x.hypot(moved.z), 0.0, 0.01);
    }
    let ball = w.add(Body::new(Shape::Sphere(0.3), 20.0, dvec3(0.0, 1.5, 6.0)));
    w.bodies[ball].vel.z = -30.0;
    run(&mut w, 10.0);
    let knocked = w.bodies[1..ball]
        .iter()
        .zip(&start[1..])
        .filter(|(b, s)| b.pos.distance(**s) > 0.5)
        .count();
    assert!(knocked >= 3, "the ball displaced only {knocked} planks");
}

#[test]
fn wall_sleeps_in_place_and_wakes_when_its_support_is_knocked_out() {
    let mut w = scenes::load(0);
    let start: Vec<Vec3> = w.bodies.iter().map(|b| b.pos).collect();
    run(&mut w, 30.0);
    for (b, s) in w.bodies.iter().zip(&start) {
        close(b.pos.distance(*s), 0.0, 0.005);
    }
    // A heavy fast ball through the bottom row: nothing may be left hanging in the air.
    let ball = w.add(Body::new(Shape::Sphere(0.3), 20.0, dvec3(0.0, 0.35, 6.0)));
    w.bodies[ball].vel.z = -30.0;
    run(&mut w, 10.0);
    let top = w.bodies[36].pos;
    assert!(top.y < 2.0, "top brick still at height {}", top.y);
    for b in &w.bodies[1..ball] {
        close(b.vel.length(), 0.0, 0.2);
    }
}

#[test]
fn sliding_box_stops_after_v_squared_over_2_mu_g() {
    let mut w = World::default();
    w.friction = 0.5;
    let b = w.add(Body::new(
        Shape::Box(Vec3::splat(0.25)),
        1.0,
        dvec3(0.0, 0.25, 0.0),
    ));
    w.bodies[b].vel.x = 3.0;
    run(&mut w, 3.0);
    close(w.bodies[b].pos.x, 9.0 / (2.0 * 0.5 * 9.81), 0.05);
    close(w.bodies[b].vel.length(), 0.0, 0.01);
}

#[test]
fn pendulum_has_the_textbook_period_and_keeps_its_energy() {
    let (r, len, angle) = (0.1, 1.0, 0.1_f64);
    let mut w = World::default();
    let bob = scenes::hang(&mut w, 0, dvec3(0.0, 3.0, 0.0), r, len, angle);

    // Same substep length as the 60 Hz default, sampled 10x more often for timing.
    w.substeps = 2;
    let (mut crossings, mut t, mut last) = (Vec::new(), 0.0, w.bodies[bob].pos.x);
    while crossings.len() < 3 {
        w.step(DT / 10.0);
        t += DT / 10.0;
        let x = w.bodies[bob].pos.x;
        if x * last < 0.0 {
            crossings.push(t - DT / 10.0 * x / (x - last));
        }
        last = x;
    }
    // 2π√(L/g) with the first finite-amplitude correction.
    let expected = std::f64::consts::TAU * (len / 9.81).sqrt() * (1.0 + angle * angle / 16.0);
    close(crossings[2] - crossings[0], expected, expected * 0.002);

    let mut w = World::default();
    let bob = scenes::hang(&mut w, 0, dvec3(0.0, 3.0, 0.0), r, len, 1.5);
    let (e0, swing) = (w.energy(), 9.81 * len / w.bodies[bob].inv_mass);
    run(&mut w, 20.0);
    close(w.energy(), e0, swing * 0.01);
}

#[test]
fn newtons_cradle_passes_the_motion_to_the_last_ball() {
    let mut w = scenes::load(2);
    let lift = w.bodies[1].pos.y - w.bodies[2].pos.y;
    let (mut last_peak, mut middle_peak) = (0.0_f64, 0.0_f64);
    for _ in 0..(2.0 / DT) as usize {
        w.step(DT);
        last_peak = last_peak.max(w.bodies[5].pos.y - w.bodies[3].pos.y);
        middle_peak = middle_peak.max(w.bodies[3].vel.length());
    }
    close(last_peak, lift, lift * 0.03);
    assert!(middle_peak < 0.05, "middle ball moved at {middle_peak} m/s");
}

#[test]
fn all_dominoes_fall() {
    let mut w = scenes::load(1);
    run(&mut w, 20.0);
    for (i, b) in w.bodies.iter().enumerate().skip(1) {
        assert!((b.rot * Vec3::Y).y < 0.5, "domino {i} is still standing");
    }
}

#[test]
fn only_the_middle_axis_spin_flips() {
    let mut w = scenes::load(4);
    let momentum = |b: &Body| b.rot * ((b.rot.inverse() * b.omega) / b.inv_inertia);
    let (e0, l0) = (w.energy(), momentum(&w.bodies[2]));
    let mut alignment = [1.0_f64; 3];
    for _ in 0..(30.0 / DT) as usize {
        w.step(DT);
        for k in 0..3 {
            let mut axis = Vec3::ZERO;
            axis[k] = 1.0;
            alignment[k] = alignment[k].min((w.bodies[k + 1].rot * axis).dot(axis));
        }
    }
    assert!(
        alignment[0] > 0.99 && alignment[2] > 0.99,
        "stable spins wandered: {alignment:?}"
    );
    assert!(
        alignment[1] < -0.9,
        "middle-axis spin never flipped: {alignment:?}"
    );
    close(w.energy(), e0, e0 * 1e-3);
    close(momentum(&w.bodies[2]).distance(l0), 0.0, l0.length() * 1e-3);
}
