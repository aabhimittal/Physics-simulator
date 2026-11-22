//! Checks the joint-coordinate robot dynamics and the demo controllers.

use glam::{DVec3 as Vec3, dvec3};
use physics_simulator::robot::{
    self, ARM_TIP, JointKind, Link, Pull, Robot, WALKER_TORSO as TORSO,
};
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
#[allow(clippy::needless_range_loop)] // reads as the matrix it checks
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
            r.q[1] > 0.7 && r.q[TORSO].abs() < 0.5,
            "fell at t = {:.1}: hip height {}, torso pitch {}",
            r.time,
            r.q[1],
            r.q[TORSO]
        );
    }
    // People walk at about 1.2 to 1.4 m/s, 1.8 to 2 steps per second, 0.6 to 0.75 m a step.
    let (speed, cadence) = (r.q[0] / r.time, r.memory[2] / r.time);
    let stride = r.q[0] / r.memory[2];
    println!("{speed:.2} m/s, {cadence:.2} steps/s, {stride:.2} m a step");
    assert!((1.0..1.5).contains(&speed), "{speed:.2} m/s");
    assert!((1.6..2.2).contains(&cadence), "{cadence:.2} steps/s");
    assert!((0.5..0.8).contains(&stride), "{stride:.2} m a step");
}

#[test]
fn walker_recovers_from_shoves_both_ways() {
    for shove in [0.06, -0.06] {
        let mut r = robot::walker();
        for frame in 0..(12.0 / DT) as usize {
            // About 250 N on the chest for 0.2 s, 5 s in.
            r.pull = (300..312).contains(&frame).then(|| Pull {
                link: TORSO,
                local: dvec3(0.0, 0.3, 0.0),
                target: dvec3(r.q[0] + shove, r.q[1] + 0.3, 0.0),
            });
            r.step(DT, 16);
            // A forward shove makes it lunge, hips down to about 0.67 m, then recover; below
            // 0.55 m it would be collapsing.
            assert!(
                r.q[1] > 0.55 && r.q[TORSO].abs() < 0.6,
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
            link: TORSO,
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
            link: TORSO,
            local: dvec3(0.0, 0.3, 0.0),
            target: dvec3(r.q[0] + shove, r.q[1] + 0.3, 0.0),
        });
        r.step(DT, 16);
        if !(r.q[1] > 0.6 && r.q[TORSO].abs() < 1.0) {
            return None;
        }
        if frame == 300 {
            (x0, steps0) = (r.q[0], r.memory[2]);
        }
        if frame < 300 {
            continue;
        }
        let down = [sole(&r, 6) < 0.005, sole(&r, 9) < 0.005];
        for (k, leg) in [(0, 4), (1, 7)] {
            if down[k] && !down[1 - k] {
                stance_knee = stance_knee.max(-r.q[leg + 1]);
            }
            if !down[k] {
                swing_knee = swing_knee.max(-r.q[leg + 1]);
            }
            if down[k] && !was_down[k] && r.q[TORSO] + r.q[leg] + r.q[leg + 1] + r.q[leg + 2] < 0.0
            {
                toe_first += 1.0;
            }
            push_off = push_off.min(r.q[leg + 2]);
        }
        was_down = down;
        tilt = tilt.max(r.q[TORSO].abs());
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

#[test]
fn walker_moves_like_the_motion_capture() {
    // Joint angles close to the recorded person's through the whole step (the hand-tuned
    // SIMBICON walk it replaced was 14.6° off), knees swinging as far as people's, heel first.
    let mut r = robot::walker();
    let (mut err, mut n) = (0.0, 0.0);
    for frame in 0..(15.0 / DT) as usize {
        r.step(DT, 16);
        if frame >= 300 {
            err += robot::imitation_error(&r);
            n += 1.0;
        }
    }
    let rms = (err / n).sqrt().to_degrees();
    let [_, _, _, swing_knee, _, toe_first, tilt] = gait_measures(0.0).expect("fell");
    println!(
        "pose {rms:.1}° rms from the capture; swing knee {:.0}°, toe-first {toe_first:.2}, tilt {:.1}°",
        swing_knee.to_degrees(),
        tilt.to_degrees()
    );
    assert!(rms < 8.0, "{rms:.1}° from the capture");
    assert!(
        (50f64.to_radians()..80f64.to_radians()).contains(&swing_knee),
        "swing knee {:.0}°",
        swing_knee.to_degrees()
    );
    assert!(toe_first < 0.2, "{toe_first:.2} of landings toe first");
    assert!(
        tilt < 10f64.to_radians(),
        "torso tilts {:.1}°",
        tilt.to_degrees()
    );
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

#[test]
fn walker_parts_match_its_links() {
    // The page draws each link by its listed part, so the list must follow the links:
    // same length, and every part attached to the part it belongs on.
    use robot::Part::*;
    let r = robot::walker();
    let parts = robot::WALKER_PARTS;
    assert_eq!(parts.len(), r.links.len());
    for (i, (part, link)) in parts.iter().zip(&r.links).enumerate() {
        let parent = link.parent.map(|p| parts[p]);
        let expected = match part {
            Hidden => {
                assert_eq!(link.mass, 0.0, "link {i} is hidden but has mass");
                continue;
            }
            Torso => Hidden,
            Thigh | UpperArm => Torso,
            Shin => Thigh,
            Foot => Shin,
            Forearm => UpperArm,
        };
        assert_eq!(
            parent,
            Some(expected),
            "link {i} ({part:?}) hangs off {parent:?}"
        );
    }
}

#[test]
fn ground_contact_stays_stable_at_one_step_per_frame() {
    // The ground springs are stiff (a 2 kg box on 30 kN/m corners bounces at about 40 Hz),
    // far too fast for explicit integration at 1/60 s. Solved implicitly, the box still
    // comes to rest on the floor at one step per frame.
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
        r.step(DT, 1);
        assert!(
            r.q[0].is_finite() && r.q[0] < 1.01,
            "blew up: height {}",
            r.q[0]
        );
    }
    close(r.q[0], 0.1, 0.002);
    close(r.qd[0], 0.0, 0.01);
}

#[test]
fn walker_stands_still_when_asked_and_shrugs_off_a_nudge() {
    // Asked for no speed, the walker stands on both feet, its ankles holding its centre of
    // mass over them; a nudge either way (about 120 N for 0.2 s) sways it but it stays put.
    for nudge in [0.0, 0.03, -0.03] {
        let mut r = robot::walker();
        r.command = 0.0;
        robot::start_in_stride(&mut r, 0.0);
        let x0 = r.q[0];
        let mut x8 = 0.0;
        for frame in 0..(10.0 / DT) as usize {
            if frame == 480 {
                x8 = r.q[0];
            }
            r.pull = (nudge != 0.0 && (240..252).contains(&frame)).then(|| Pull {
                link: TORSO,
                local: dvec3(0.0, 0.3, 0.0),
                target: dvec3(r.q[0] + nudge, r.q[1] + 0.3, 0.0),
            });
            r.step(DT, 16);
            assert!(r.q[1] > 0.7, "nudge {nudge}: fell at t = {:.1}", r.time);
        }
        assert_eq!(r.memory[4], 1.0, "not standing");
        assert!(
            (r.q[0] - x0).abs() < 0.3,
            "nudge {nudge}: wandered {:.2} m",
            r.q[0] - x0
        );
        // Settled: its feet grip rather than creep (measured over the last 2 s, as the joint
        // springs keep up a buzz of a few micrometres that a single reading would catch).
        let creep = (r.q[0] - x8) / 2.0;
        assert!(
            creep.abs() < 0.01,
            "nudge {nudge}: still moving at {creep:.3} m/s"
        );
    }
}

#[test]
fn walker_follows_the_pace_it_is_asked_for() {
    // From a slow stroll to a brisk walk: the speed it settles to matches the request.
    for pace in [0.8, 1.36, 1.9] {
        let mut r = robot::walker();
        r.command = pace;
        robot::start_in_stride(&mut r, 0.0);
        let mut x5 = 0.0;
        for frame in 0..(12.0 / DT) as usize {
            r.step(DT, 16);
            assert!(r.q[1] > 0.7, "pace {pace}: fell at t = {:.1}", r.time);
            if frame == 300 {
                x5 = r.q[0];
            }
        }
        let speed = (r.q[0] - x5) / 7.0;
        println!("asked {pace:.2} m/s, walked {speed:.2}");
        assert!(
            (speed - pace).abs() < 0.2,
            "asked {pace:.2} m/s, walked {speed:.2}"
        );
    }
}

#[test]
fn walker_stops_and_sets_off_again() {
    // Walking, then asked to stop (the request easing off at 0.8 m/s², as the crossing does),
    // then to walk on: it slows, stands on both feet, and steps off again without falling.
    let mut r = robot::walker();
    let (mut stood, mut walked_again) = (false, false);
    for frame in 0..(16.0 / DT) as usize {
        let wanted = if (180..480).contains(&frame) {
            0.0
        } else {
            robot::WALKING_SPEED
        };
        r.command += (wanted - r.command).clamp(-0.8 * DT, 0.8 * DT);
        r.step(DT, 16);
        assert!(r.q[1] > 0.7, "fell at t = {:.1}", r.time);
        stood |= frame == 470 && r.memory[4] == 1.0 && r.qd[0].abs() < 0.1;
        walked_again |= frame > 800 && r.qd[0] > 1.0;
    }
    assert!(stood, "had not come to a stand by t = 7.8 s");
    assert!(walked_again, "did not walk on");
}
