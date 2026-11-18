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

#[test]
fn dropped_box_comes_to_rest_on_the_ground() {
    let half = dvec3(0.2, 0.1, 0.2);
    let mut r = Robot::new(vec![Link::new(
        None,
        JointKind::Slide,
        Vec3::Y,
        Vec3::ZERO,
        Vec3::ZERO,
        half,
        2.0,
    )]);
    r.ground = true;
    r.q[0] = 1.0;
    for _ in 0..(3.0 / DT) as usize {
        r.step(DT, 16);
    }
    close(r.q[0], 0.1, 0.002);
    close(r.qd[0], 0.0, 0.01);
}

#[test]
fn hopper_travels_between_its_marks_without_falling() {
    let mut r = robot::hopper();
    let (mut hops, mut was_up) = (0, false);
    for frame in 0..(25.0 / DT) as usize {
        // A sideways shove on the body partway through (about 1 m/s) must not topple it.
        r.pull = (frame / 60 == 12 && frame % 60 < 18).then(|| Pull {
            link: 2,
            local: Vec3::ZERO,
            target: dvec3(r.q[0] - 0.05, r.q[1], 0.0),
        });
        r.step(DT, 16);
        assert!(
            r.q[1] > 0.7 && r.q[2].abs() < 0.6,
            "fell at t = {}: height {}, pitch {}",
            r.time,
            r.q[1],
            r.q[2]
        );
        let up = r.q[1] > 1.02;
        hops += (was_up && !up) as u32;
        was_up = up;
        // The goal is x = +2 for the first 10 s, then x = -2 for the next 10.
        if frame == 10 * 60 {
            assert!(r.q[0] > 1.5, "only reached x = {} going right", r.q[0]);
        }
        if frame == 20 * 60 {
            assert!(r.q[0] < -1.5, "only reached x = {} going left", r.q[0]);
        }
    }
    assert!(hops >= 30, "only {hops} hops in 25 s");
}

#[test]
fn joint_limit_stops_a_motor_driving_past_it() {
    let mut slider = Link::new(
        None,
        JointKind::Slide,
        Vec3::X,
        Vec3::ZERO,
        Vec3::ZERO,
        dvec3(0.1, 0.1, 0.1),
        1.0,
    );
    (slider.gear, slider.range, slider.damping) = (50.0, Some((-0.5, 0.5)), 5.0);
    let mut r = Robot::new(vec![slider]);
    r.gravity = Vec3::ZERO;
    r.ctrl[0] = 1.0;
    run(&mut r, 3.0);
    // Rests where the limit spring balances the motor: 50 N / 20000 N/m past the limit.
    close(r.q[0], 0.5 + 50.0 / 2.0e4, 1e-3);
}

#[test]
fn motor_force_falls_to_zero_at_its_free_speed() {
    let mut slider = Link::new(
        None,
        JointKind::Slide,
        Vec3::X,
        Vec3::ZERO,
        Vec3::ZERO,
        dvec3(0.1, 0.1, 0.1),
        1.0,
    );
    (slider.gear, slider.free_speed) = (50.0, 2.0);
    let mut r = Robot::new(vec![slider]);
    r.gravity = Vec3::ZERO;
    r.ctrl[0] = 1.0;
    // From rest the full 50 N acts on 1 kg; at half the free speed, half of it.
    close(r.accelerations(&[0.0], &[0.0])[0], 50.0, 1e-9);
    close(r.accelerations(&[0.0], &[1.0])[0], 25.0, 1e-9);
    // Braking is not weakened by speed, and above the free speed the motor can only brake.
    r.ctrl[0] = -1.0;
    close(r.accelerations(&[0.0], &[1.0])[0], -50.0, 1e-9);
    r.ctrl[0] = 1.0;
    close(r.accelerations(&[0.0], &[3.0])[0], -25.0, 1e-9);
    close(r.accelerations(&[0.0], &[9.0])[0], -50.0, 1e-9);
    // Driven flat out with nothing in the way, it approaches the free speed and stays there.
    run(&mut r, 2.0);
    close(r.qd[0], 2.0, 1e-6);
}

#[test]
fn walker_walks_at_human_speed_and_cadence() {
    let mut r = robot::walker();
    for _ in 0..(15.0 / DT) as usize {
        r.step(DT, 16);
        assert!(
            r.q[1] > 0.7 && r.q[2].abs() < 0.5,
            "fell at t = {:.1}: hip height {}, torso pitch {}",
            r.time,
            r.q[1],
            r.q[2]
        );
    }
    // A brisk human walk: about 1.5 to 1.8 m/s at 1.7 to 2 steps per second.
    let (speed, cadence) = (r.q[0] / r.time, r.memory[2] / r.time);
    let stride = r.q[0] / r.memory[2];
    println!("{speed:.2} m/s, {cadence:.2} steps/s, {stride:.2} m a step");
    assert!((1.3..1.8).contains(&speed), "{speed:.2} m/s");
    assert!((1.6..2.1).contains(&cadence), "{cadence:.2} steps/s");
    assert!((0.7..1.0).contains(&stride), "{stride:.2} m a step");
}

#[test]
fn walker_recovers_from_shoves_both_ways() {
    for shove in [0.06, -0.06] {
        let mut r = robot::walker();
        for frame in 0..(12.0 / DT) as usize {
            // About 250 N on the chest for 0.2 s, 5 s in.
            r.pull = (300..312).contains(&frame).then(|| Pull {
                link: 2,
                local: dvec3(0.0, 0.3, 0.0),
                target: dvec3(r.q[0] + shove, r.q[1] + 0.3, 0.0),
            });
            r.step(DT, 16);
            // A forward shove makes it lunge, hips down to about 0.67 m, then recover; below
            // 0.55 m it would be collapsing.
            assert!(
                r.q[1] > 0.55 && r.q[2].abs() < 0.6,
                "shove {shove}: fell at t = {:.1}",
                r.time
            );
        }
        assert!(
            r.qd[0] > 0.3 || r.q[0] > 8.0,
            "shove {shove}: stopped walking forward"
        );
    }
}

#[test]
fn knocked_down_walker_comes_to_rest() {
    // A hard shove knocks it over. The fallen body must stop, not slide off faster and
    // faster (stiff joint dampers on the light feet once made it gain momentum from nowhere).
    let mut r = robot::walker();
    for frame in 0..(12.0 / DT) as usize {
        r.pull = (240..249).contains(&frame).then(|| Pull {
            link: 2,
            local: dvec3(0.0, 0.3, 0.0),
            target: dvec3(r.q[0] + 1.0, r.q[1] + 0.3, 0.0),
        });
        r.step(DT, 16);
    }
    assert!(r.q.iter().chain(&r.qd).all(|v| v.is_finite()));
    assert!(r.qd[0].abs() < 0.2, "still sliding at {} m/s", r.qd[0]);
}

/// Gait measurements over seconds 5 to 15 of walking: speed, cadence, the most the knee of
/// the only foot on the ground bends, the swing knee's peak, the ankle's push-off (most
/// toe-down), toe-first landings per step, and the torso's largest tilt. `None` if it fell.
fn gait_measures(shove: f64) -> Option<[f64; 7]> {
    let sole = |r: &Robot, foot: usize| {
        let (o, rot) = r.frames(&r.q)[foot];
        let l = &r.links[foot];
        (0..4)
            .map(|k| {
                (o + rot * (l.com + l.half * dvec3([-1.0, 1.0][k & 1], -1.0, [-1.0, 1.0][k >> 1])))
                    .y
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut r = robot::walker();
    let (mut stance_knee, mut swing_knee, mut push_off, mut tilt, mut toe_first) =
        (0f64, 0f64, 0f64, 0f64, 0.0);
    let (mut was_down, mut x0, mut steps0) = ([true, true], 0.0, 0.0);
    for frame in 0..(15.0 / DT) as usize {
        r.pull = (shove != 0.0 && (300..312).contains(&frame)).then(|| Pull {
            link: 2,
            local: dvec3(0.0, 0.3, 0.0),
            target: dvec3(r.q[0] + shove, r.q[1] + 0.3, 0.0),
        });
        r.step(DT, 16);
        if !(r.q[1] > 0.6 && r.q[2].abs() < 1.0) {
            return None;
        }
        if frame == 300 {
            (x0, steps0) = (r.q[0], r.memory[2]);
        }
        if frame < 300 {
            continue;
        }
        let down = [sole(&r, 5) < 0.005, sole(&r, 8) < 0.005];
        for (k, leg) in [(0, 3), (1, 6)] {
            if down[k] && !down[1 - k] {
                stance_knee = stance_knee.max(-r.q[leg + 1]);
            }
            if !down[k] {
                swing_knee = swing_knee.max(-r.q[leg + 1]);
            }
            if down[k] && !was_down[k] && r.q[2] + r.q[leg] + r.q[leg + 1] + r.q[leg + 2] < 0.0 {
                toe_first += 1.0;
            }
            push_off = push_off.min(r.q[leg + 2]);
        }
        was_down = down;
        tilt = tilt.max(r.q[2].abs());
    }
    let steps = r.memory[2] - steps0;
    Some([
        (r.q[0] - x0) / 10.0,
        steps / 10.0,
        stance_knee,
        swing_knee,
        push_off,
        toe_first / steps.max(1.0),
        tilt,
    ])
}

/// How far a gait is from typical adult walking (speed 1.3 m/s, 1.85 steps a second, stance
/// knee 18°, swing knee 60°, push-off to -16°, heel-first landings, upright torso), and
/// heavily penalised if it falls on its own or after a 250 N shove either way.
fn gait_cost() -> f64 {
    let Some(
        [
            speed,
            cadence,
            stance_knee,
            swing_knee,
            push_off,
            toe_first,
            tilt,
        ],
    ) = gait_measures(0.0)
    else {
        return 1e4;
    };
    let d = f64::to_radians;
    let falls = [0.06, -0.06]
        .iter()
        .filter(|&&s| gait_measures(s).is_none())
        .count() as f64;
    ((speed - 1.3) / 0.1).powi(2)
        + ((cadence - 1.85) / 0.1).powi(2)
        + ((stance_knee - d(18.0)) / d(5.0)).powi(2)
        + ((swing_knee - d(60.0)) / d(10.0)).powi(2)
        + ((push_off + d(16.0)) / d(4.0)).powi(2)
        + 10.0 * toe_first
        + (tilt / d(8.0)).powi(2)
        + 100.0 * falls
}

/// Tunes the walking gait by random search: perturb the best parameters, keep a trial if it
/// scores better, widen the search after a success and narrow it after failures. Prints the
/// best gait as Rust to paste into `robot::WALK`. Slow, so opt in:
/// `cargo test --release --test robot tune_walk -- --ignored --nocapture`.
#[test]
#[ignore]
fn tune_walk() {
    use physics_simulator::env::Rng;
    use physics_simulator::robot::{Gait, WALK, with_gait};
    let pack = |g: &Gait| {
        [
            g.swing_time,
            g.swinging.swing_hip,
            g.swinging.swing_knee,
            g.swinging.swing_ankle,
            g.swinging.stance_knee,
            g.landing.swing_hip,
            g.landing.swing_knee,
            g.landing.swing_ankle,
            g.landing.stance_knee,
            g.distance_gain,
            g.speed_gain,
            g.lean,
            g.push_off,
            g.knee_stiffness,
            g.ankle_stiffness,
        ]
    };
    let unpack = |v: &[f64; 15]| {
        let mut g = WALK;
        [
            g.swing_time,
            g.swinging.swing_hip,
            g.swinging.swing_knee,
            g.swinging.swing_ankle,
            g.swinging.stance_knee,
            g.landing.swing_hip,
            g.landing.swing_knee,
            g.landing.swing_ankle,
            g.landing.stance_knee,
            g.distance_gain,
            g.speed_gain,
            g.lean,
            g.push_off,
            g.knee_stiffness,
            g.ankle_stiffness,
        ] = *v;
        g
    };
    let scale = [
        0.04, 0.08, 0.1, 0.1, 0.05, 0.08, 0.05, 0.08, 0.05, 0.08, 0.05, 0.02, 0.08, 60.0, 40.0,
    ];
    let score = |v: &[f64; 15]| with_gait(unpack(v), gait_cost);
    let mut best = pack(&WALK);
    let mut best_cost = score(&best);
    let mut rng = Rng::new(7);
    let mut sigma = 1.0;
    println!("start: cost {best_cost:.1}");
    for round in 0..120 {
        let trials: Vec<[f64; 15]> = (0..4)
            .map(|_| std::array::from_fn(|i| best[i] + sigma * scale[i] * rng.uniform()))
            .collect();
        let costs: Vec<f64> = std::thread::scope(|s| {
            let handles: Vec<_> = trials.iter().map(|t| s.spawn(move || score(t))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (i, &cost) = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        if cost < best_cost {
            (best, best_cost) = (trials[i], cost);
            sigma = (sigma * 1.3).min(2.0);
            println!(
                "round {round}: cost {best_cost:.2} {:?}",
                with_gait(unpack(&best), || gait_measures(0.0))
            );
        } else {
            sigma = (sigma * 0.85).max(0.1);
        }
    }
    println!("best cost {best_cost:.2}\n{:#?}", unpack(&best));
}

#[test]
fn walker_steps_evenly() {
    // Every step alike, as in a person's walk: an irregular gait (long steps building up to a
    // stumble) is what looked most robotic. Step lengths after 5 s stay within 10% of their mean.
    let mut r = robot::walker();
    let (mut lengths, mut last_x, mut steps) = (vec![], 0.0, 0.0);
    for frame in 0..(25.0 / DT) as usize {
        r.step(DT, 16);
        if r.memory[2] != steps {
            if frame > 300 {
                lengths.push(r.q[0] - last_x);
            }
            (last_x, steps) = (r.q[0], r.memory[2]);
        }
    }
    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    for (i, l) in lengths.iter().enumerate() {
        assert!(
            (l - mean).abs() < 0.1 * mean,
            "step {i} is {l:.2} m against a mean of {mean:.2} m"
        );
    }
}
