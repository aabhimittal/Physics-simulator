//! Checks the car against textbook vehicle dynamics.

use glam::DVec3 as Vec3;
use physics_simulator::robot;
use physics_simulator::scenes::{self, CAR_SCENE};
use physics_simulator::vehicle::{Car, WHEELBASE};
use physics_simulator::{Body, Shape, World};

const DT: f64 = 1.0 / 60.0;

/// The car alone on flat ground, moving straight ahead at `speed` with its wheels rolling.
fn car_at(speed: f64) -> World {
    let mut w = scenes::load(CAR_SCENE);
    w.bodies.truncate(2); // ground and car body
    let car = &mut w.cars[0];
    car.autopilot = None;
    for wheel in &mut car.wheels {
        wheel.spin = speed / 0.33;
    }
    w.bodies[1].vel = Vec3::X * speed;
    w
}

#[test]
fn parked_car_rests_at_its_ride_height() {
    let mut w = car_at(0.0);
    for _ in 0..(5.0 / DT) as usize {
        w.step(DT);
    }
    let b = &w.bodies[1];
    assert!(
        (b.pos.y - physics_simulator::vehicle::Car::RIDE_HEIGHT).abs() < 2e-3,
        "height {}",
        b.pos.y
    );
    assert!(
        b.vel.length() < 0.01,
        "still moving at {} m/s",
        b.vel.length()
    );
}

#[test]
fn full_throttle_reaches_100_kmh_in_a_plausible_time() {
    let mut w = car_at(0.0);
    let mut t = 0.0;
    while w.bodies[1].vel.x < 100.0 / 3.6 {
        assert!(
            t < 12.0,
            "only {:.0} km/h after 12 s",
            w.bodies[1].vel.x * 3.6
        );
        w.cars[0].throttle = 1.0;
        w.step(DT);
        t += DT;
    }
    // 1200 kg and 90 kW: about 8 to 10 s, like a small hatchback.
    assert!(
        t > 7.0,
        "0-100 km/h in {t:.1} s is faster than 90 kW allows"
    );
}

#[test]
fn abs_braking_stops_in_the_distance_friction_allows() {
    let v0: f64 = 100.0 / 3.6;
    let mut w = car_at(v0);
    let x0 = w.bodies[1].pos.x;
    while w.bodies[1].vel.x > 0.05 {
        w.cars[0].brake = 1.0;
        w.step(DT);
        if w.bodies[1].vel.x > 1.0 {
            assert!(
                w.cars[0].wheels.iter().all(|wh| wh.spin > 0.0),
                "a wheel locked"
            );
        }
    }
    // With friction coefficient 1 the shortest stop is v² / 2g = 39.3 m.
    let distance = w.bodies[1].pos.x - x0;
    let ideal = v0 * v0 / (2.0 * 9.81);
    assert!(
        distance > ideal * 0.98 && distance < ideal * 1.1,
        "stopped in {distance:.1} m, ideal {ideal:.1} m"
    );
}

#[test]
fn autopilot_corners_on_its_circle_with_the_right_sideways_acceleration() {
    let mut w = scenes::load(CAR_SCENE);
    let (mut last_bump, mut checked) = (0, 0);
    for frame in 0..(40.0 / DT) as usize {
        w.step(DT);
        let (b, car) = (&w.bodies[w.cars[0].body], &w.cars[0]);
        let off = (b.pos.x.hypot(b.pos.z) - 30.0).abs();
        assert!(
            off < 1.0,
            "{off:.2} m off the circle at t = {:.1}",
            car.time
        );
        // The speed hump on the circle jolts the readout; measure away from it.
        if car
            .wheels
            .iter()
            .any(|wheel| wheel.ground.is_some_and(|g| g != 0))
        {
            last_bump = frame;
        }
        // At steady speed the sideways acceleration is v² / r: once per 16 s lap of the
        // autopilot's speed cycle, at least half a second clear of the hump.
        if frame % 960 >= 420 && frame / 960 >= checked && frame - last_bump > 30 {
            checked = frame / 960 + 1;
            let expected = b.vel.length_squared() / 30.0;
            let measured = car.accel.length();
            assert!(
                (measured - expected).abs() < 0.15 * expected,
                "{measured:.2} m/s² sideways, expected {expected:.2}"
            );
            assert!(
                car.roll(b).abs() > 0.02,
                "no body roll in a {expected:.1} m/s² corner"
            );
            // Anti-roll bars keep it under 4°, about what a real small car leans at 0.7 g.
            let roll = car.roll(b).abs().to_degrees();
            assert!(roll < 4.0, "{roll:.1}° of body roll");
        }
    }
    assert_eq!(checked, 3, "measured on {checked} laps of 3");
}

#[test]
fn car_stops_for_people_crossing_and_drives_on() {
    let (mut w, mut walker) = scenes::crossing();
    let (mut closest, mut stopped, mut driven, mut crossed) = (f64::INFINITY, 0, 0.0, 0);
    let mut was_stopped = false;
    for _ in 0..(60.0 / DT) as usize {
        let before = walker.q[0];
        scenes::step_crossing(&mut w, &mut walker, DT);
        crossed += (walker.q[0] < before - 5.0) as u32; // replaced at the kerb
        let (car, b) = (&w.cars[0], &w.bodies[1]);
        let person = Vec3::new(walker.q[0], 0.0, 0.0);
        if ((person.x.hypot(person.z)) - 30.0).abs() < 3.5 {
            closest = closest.min(b.pos.with_y(0.0).distance(person));
        }
        let stopped_now = car.yielding && b.vel.length() < 0.5;
        stopped += (stopped_now && !was_stopped) as u32;
        was_stopped = stopped_now;
        driven += b.vel.length() * DT;
        assert!(walker.q[1] > 0.7, "the person fell at t = {:.1}", car.time);
    }
    println!(
        "closest {closest:.1} m (car centre to person on the road), stopped {stopped} times, drove {driven:.0} m, {crossed} people crossed"
    );
    // The car's nose is 2.1 m ahead of its centre: at least 1.5 m clear at the closest.
    assert!(
        closest > 3.6,
        "car centre came within {closest:.1} m of the person on the road"
    );
    assert!(stopped >= 2, "stopped for people only {stopped} times");
    assert!(crossed >= 3, "only {crossed} people got across");
    assert!(
        driven > 250.0,
        "the car only drove {driven:.0} m in a minute"
    );
}

#[test]
fn gentle_throttle_shifts_up_early_and_braking_does_not_drop_to_first() {
    // An automatic picks its shift points by how hard the driver presses: gently, it changes
    // up at low revs. It used to wait for 6200 rpm whatever the throttle, so the autopilot
    // cruised at 50 km/h in first gear.
    let mut w = car_at(8.0);
    while w.bodies[1].vel.x < 15.0 {
        w.cars[0].throttle = 0.3;
        w.step(DT);
        assert!(w.cars[0].time < 30.0, "never reached 15 m/s");
    }
    let car = &w.cars[0];
    assert!(
        car.gear >= 3,
        "gear {} at 15 m/s on light throttle",
        car.gear
    );
    assert!(car.rpm < 3500.0, "{:.0} rpm on light throttle", car.rpm);
    // Braking from 25 m/s: the box follows the road speed down a gear at a time, never
    // revving the engine up through a low gear.
    let mut w = car_at(25.0);
    w.cars[0].gear = 5;
    while w.bodies[1].vel.x > 10.0 {
        w.cars[0].brake = 0.6;
        w.step(DT);
        assert!(
            w.cars[0].rpm < 3000.0,
            "{:.0} rpm while braking",
            w.cars[0].rpm
        );
    }
    assert!(w.cars[0].gear >= 3, "gear {} at 10 m/s", w.cars[0].gear);
}

#[test]
fn gearbox_shifts_up_as_the_car_speeds_up() {
    let mut w = car_at(0.0);
    let mut gears = vec![1];
    while w.bodies[1].vel.x < 120.0 / 3.6 {
        w.cars[0].throttle = 1.0;
        w.step(DT);
        let car = &w.cars[0];
        if car.gear != *gears.last().unwrap() {
            gears.push(car.gear);
        }
        assert!(
            car.rpm <= 6900.0,
            "engine at {:.0} rpm, past the limiter",
            car.rpm
        );
        assert!(
            car.time < 30.0,
            "only {:.0} km/h after 30 s",
            w.bodies[1].vel.x * 3.6
        );
    }
    // One gear at a time, upward only, and well into the box by 120 km/h.
    assert!(
        gears.windows(2).all(|g| g[1] == g[0] + 1),
        "shifted {gears:?}"
    );
    assert!(
        *gears.last().unwrap() >= 3,
        "still in gear {} at 120 km/h",
        gears.last().unwrap()
    );
}

#[test]
fn a_wheel_on_a_block_stands_on_it() {
    // A 10 cm block under the front left wheel: that wheel's ray finds the block, the
    // others the ground, and the body leans away from the block's corner.
    let mut w = car_at(0.0);
    let block = w.add(Body::new(
        Shape::Box(Vec3::new(0.3, 0.05, 0.3)),
        0.0,
        Vec3::new(1.35, 0.05, 30.8),
    ));
    for _ in 0..(3.0 / DT) as usize {
        w.step(DT);
    }
    let (car, b) = (&w.cars[0], &w.bodies[1]);
    assert_eq!(car.wheels[0].ground, Some(block));
    assert!(car.wheels[1..].iter().all(|wheel| wheel.ground == Some(0)));
    let up = b.rot * Vec3::Y;
    assert!(
        up.x < -0.005 && up.z < -0.005,
        "body not tilted away from the block: up = {up:?}"
    );
    assert!(b.vel.length() < 0.05, "not at rest on the block");
}

#[test]
fn wheels_roll_over_a_kerb_instead_of_snapping_onto_it() {
    // A 10 cm kerb across the road, met at 2 m/s. A tyre meets the kerb's corner ahead of its
    // lowest point and rolls up over it: the wheel rises about 1 cm per cm of travel at
    // first, a few cm a frame. A single ray straight down would find the kerb all at once and
    // jolt the wheel up the whole 10 cm in one step.
    let mut w = car_at(2.0);
    w.add(Body::new(
        Shape::Box(Vec3::new(1.0, 0.05, 3.0)),
        0.0,
        Vec3::new(5.0, 0.05, 30.0),
    ));
    let (mut jump, mut last) = (0f64, w.cars[0].wheels[0].length);
    for _ in 0..(5.0 / DT) as usize {
        w.cars[0].throttle = 0.25;
        w.step(DT);
        let wheel = &w.cars[0].wheels[0];
        jump = jump.max((last - wheel.length).abs());
        last = wheel.length;
    }
    println!(
        "biggest suspension change in a frame {:.1} cm",
        jump * 100.0
    );
    assert!(
        w.bodies[1].pos.x > 7.0,
        "did not get over the kerb (x = {:.1})",
        w.bodies[1].pos.x
    );
    assert!(
        jump < 0.05,
        "wheel jumped {:.1} cm in one frame",
        jump * 100.0
    );
}

#[test]
fn a_box_beside_the_tyres_does_not_lift_them() {
    // A 10 cm block along the car's left side, its edge 0.2 m outside the left tyres'
    // centre line: beyond the tyre's tread, so the wheels stay on the road. (A swept sphere
    // as wide as the tyre is tall would touch the block's edge and climb onto it.)
    let mut w = car_at(0.0);
    w.add(Body::new(
        Shape::Box(Vec3::new(3.0, 0.05, 0.5)),
        0.0,
        Vec3::new(0.0, 0.05, 30.8 + 0.2 + 0.5),
    ));
    for _ in 0..(2.0 / DT) as usize {
        w.step(DT);
        assert!(
            w.cars[0].wheels.iter().all(|wheel| wheel.ground == Some(0)),
            "a wheel touches the block beside it at t = {:.2}",
            w.cars[0].time
        );
    }
    let shoved = (w.bodies[1].pos.z - 30.0).abs();
    assert!(shoved < 0.01, "pushed {shoved:.3} m sideways by the block");
}

#[test]
fn car_drives_up_a_ramp_jumps_off_and_lands_upright() {
    // A 4 m ramp rising to 0.6 m (8.6°) across the car's path.
    let mut w = car_at(8.0);
    let mut ramp = Body::new(
        Shape::Box(Vec3::new(2.0, 0.3, 1.5)),
        0.0,
        Vec3::new(10.0, 0.0, 30.0),
    );
    ramp.rot = glam::DQuat::from_rotation_z(0.15);
    w.add(ramp);
    let (mut highest, mut airborne) = (0f64, false);
    for _ in 0..(6.0 / DT) as usize {
        w.cars[0].throttle = 0.3;
        w.step(DT);
        highest = highest.max(w.bodies[1].pos.y);
        airborne |= w.cars[0].wheels.iter().all(|wheel| wheel.load == 0.0);
    }
    let (car, b) = (&w.cars[0], &w.bodies[1]);
    assert!(
        highest > Car::RIDE_HEIGHT + 0.5,
        "only rose to {highest:.2} m"
    );
    assert!(airborne, "never left the ground off the end of the ramp");
    assert!(
        b.pos.x > 25.0 && b.vel.x > 3.0,
        "stopped at x = {:.1}",
        b.pos.x
    );
    let upright = (b.rot * Vec3::Y).y;
    assert!(
        upright > 0.99 && car.wheels.iter().all(|wheel| wheel.ground == Some(0)),
        "did not land on its wheels: up.y {upright:.3}"
    );
}

/// Brakes failed: no autopilot, coasting at 8 m/s down the road (along -z at x = 30) for 3 s,
/// with the walker starting at `person_x`. Returns the car's final speed, the person's
/// fastest speed along the car's way, and how far they were carried.
fn coast_into_person(person_x: f64) -> (f64, f64, f64) {
    let (mut w, mut walker) = scenes::crossing();
    let car = &mut w.cars[0];
    car.autopilot = None;
    for wheel in &mut car.wheels {
        wheel.spin = 8.0 / 0.33;
    }
    let b = &mut w.bodies[1];
    let ahead = glam::DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
    (b.pos, b.rot, b.vel) = (
        Vec3::new(30.0, b.pos.y, 12.0),
        ahead,
        Vec3::new(0.0, 0.0, -8.0),
    );
    walker.q[0] = person_x;
    robot::start_in_stride(&mut walker, 0.0); // on the road surface, not the pavement's height
    let mut fastest = 0f64;
    for _ in 0..(3.0 / DT) as usize {
        scenes::step_crossing(&mut w, &mut walker, DT);
        assert!(
            walker.q.iter().chain(&walker.qd).all(|v| v.is_finite()),
            "the person's state blew up"
        );
        fastest = fastest.max(-walker.qd[2]);
    }
    (-w.bodies[1].vel.z, fastest, -walker.q[2])
}

#[test]
fn a_car_that_does_not_stop_hits_the_person_and_both_feel_it() {
    let (missed, _, _) = coast_into_person(100.0); // nobody in the road: drag and rolling only
    // The person walks at 1.25 m/s from the start, so starting 1.8 m short of the car's line
    // puts them in its way as it arrives (any start from 27.8 to 28.6 m hits).
    let (hit, thrown, carried) = coast_into_person(28.6);
    println!(
        "car ends at {hit:.2} m/s (at {missed:.2} with nobody there); person thrown at up to {thrown:.1} m/s, carried {carried:.1} m"
    );
    // The person is knocked along the car's way, about as fast as the car was going.
    assert!(
        thrown > 4.0 && thrown < 12.0,
        "person thrown at {thrown:.1} m/s"
    );
    assert!(carried > 2.0, "person only moved {carried:.1} m");
    // And the car pays for it: the momentum it lost beyond coasting went into the person
    // (some of it on into the ground, through the person's feet).
    let lost = 1200.0 * (missed - hit);
    assert!(
        lost > 70.0 * thrown * 0.7,
        "car lost only {lost:.0} kg·m/s to a {:.0} kg·m/s throw",
        70.0 * thrown
    );
}

#[test]
fn with_its_brakes_failed_the_car_hits_the_person_crossing() {
    // The same driver sees the person and lifts off, but cannot stop in time.
    let (mut w, mut walker) = scenes::crossing_brakes_failed();
    let mut impact = None;
    for frame in 0..(10.0 / DT) as usize {
        scenes::step_crossing(&mut w, &mut walker, DT);
        assert!(
            walker.q.iter().chain(&walker.qd).all(|v| v.is_finite()),
            "the person's state blew up"
        );
        let speed = w.bodies[1].vel.length();
        if impact.is_none() && walker.qd[2].abs() > 2.0 {
            impact = Some((frame as f64 * DT, speed * 3.6));
        }
    }
    let (at, kmh) = impact.expect("the car never hit the person");
    println!("impact at {at:.1} s, car at {kmh:.0} km/h");
    assert!(
        kmh > 30.0,
        "only {kmh:.0} km/h at impact: it nearly stopped"
    );
}

#[test]
fn the_driver_never_speeds_up_toward_someone_in_the_way() {
    // The ordinary crossing starts the car slow (8 m/s), below its cruising speed: once it
    // has seen the person it must lift off, not keep accelerating up to its braking limit.
    let (mut w, mut walker) = scenes::crossing();
    let (mut seen_at, mut fastest_after) = (None, 0f64);
    for _ in 0..(4.0 / DT) as usize {
        scenes::step_crossing(&mut w, &mut walker, DT);
        let speed = w.bodies[1].vel.length();
        if w.cars[0].yielding {
            seen_at.get_or_insert(speed);
            fastest_after = fastest_after.max(speed);
        }
    }
    let seen_at = seen_at.expect("the driver never saw the person");
    assert!(
        fastest_after < seen_at + 0.05,
        "sped up from {seen_at:.2} to {fastest_after:.2} m/s toward someone in its way"
    );
}

/// Slope of `y` against `x` (least squares, with an offset).
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    sxy / sxx
}

#[test]
fn turns_like_a_real_civic_on_recorded_drives() {
    // Twelve one-minute drives of a 2016 Honda Civic from comma2k19 (python/comma2k19.py),
    // none of them used to choose the car's settings. Each is replayed: the recorded steering
    // through the Civic's 15.38:1 steering ratio, and throttle and brake following the
    // recorded speed. The yaw rate that results is compared with the phone gyro's.
    const RATIO: f64 = 15.38;
    let rows: Vec<[f64; 5]> = include_str!("data/comma2k19.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|s| s.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4]]
        })
        .collect();
    let (mut gain_errors, mut tyres, mut geometry) = (vec![], 0.0, 0.0);
    for drive in rows.chunk_by(|a, b| a[0] == b[0]) {
        let mut w = car_at(drive[0][2]);
        let mut integral = 0.0;
        let (mut kinematic, mut real, mut sim, mut speed) = (vec![], vec![], vec![], 0.0);
        for (now, next) in drive.iter().zip(&drive[1..]) {
            for _ in 0..6 {
                // A cruise controller holding the recorded speed.
                let b = &w.bodies[1];
                let error = now[2] - b.vel.dot(b.rot * Vec3::X);
                integral = (integral + error * DT).clamp(-5.0, 5.0);
                let push = 0.5 * error + 0.2 * integral;
                let car = &mut w.cars[0];
                (car.throttle, car.brake) = (push.clamp(0.0, 1.0), (-push).clamp(0.0, 1.0));
                car.steer = now[3].to_radians() / RATIO;
                w.step(DT);
            }
            if next[2] > 3.0 {
                let wheels = now[3].to_radians() / RATIO;
                kinematic.push(now[2] * wheels.tan() / WHEELBASE);
                real.push(next[4]);
                sim.push(w.bodies[1].omega.y);
                speed += now[2];
            }
        }
        let n = real.len() as f64;
        let rms = |a: &[f64]| {
            (a.iter()
                .zip(&real)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
        };
        let (real_gain, sim_gain) = (slope(&kinematic, &real), slope(&kinematic, &sim));
        println!(
            "drive {}: {:.1} m/s, yaw gain {real_gain:.2} recorded, {sim_gain:.2} simulated; yaw error {:.4} rad/s (no-slip geometry: {:.4})",
            drive[0][0],
            speed / n,
            rms(&sim),
            rms(&kinematic)
        );
        if speed / n > 20.0 {
            gain_errors.push(sim_gain - real_gain);
            (tyres, geometry) = (tyres + rms(&sim), geometry + rms(&kinematic));
        }
    }
    // On the highway the Civic turns only about 0.65 times as tightly as its wheels point
    // (understeer); the car must match that, drive by drive.
    let mean = gain_errors.iter().sum::<f64>() / gain_errors.len() as f64;
    assert!(mean.abs() < 0.05, "yaw gain off by {mean:.3} on average");
    assert!(
        gain_errors.iter().all(|e| e.abs() < 0.1),
        "{gain_errors:.2?}"
    );
    // And the tyre model must predict the yaw rate better than steering geometry alone.
    assert!(tyres < 0.8 * geometry, "{tyres:.3} against {geometry:.3}");
}

#[test]
fn coasts_down_like_the_epas_civic() {
    // Every US car is coast-down tested: from speed, in neutral, on a flat track, and the
    // deceleration gives its road load F(v) = A + B·v + C·v². The EPA's 2016 Test Car List
    // has the 2016 Civic 4-door, 1.5 L CVT (the comma2k19 car's drivetrain) at
    // A = 30.64 lbf, B = -0.0741 lbf/mph, C = 0.01821 lbf/mph², test weight 3250 lb.
    // Rolling resistance scales with weight, so compare it as a coefficient, A / (m·g);
    // drag does not, so compare C as a force.
    let (lbf, mph) = (4.448_222, 0.447_04);
    let civic_rolling = 30.64 * lbf / (3250.0 * 0.453_592 * 9.81);
    let civic_drag = 0.01821 * lbf / (mph * mph);
    // The same test on the car: coast from 35 m/s with no throttle, and fit F = m·a.
    let mut w = car_at(35.0);
    let mass = 1.0 / w.bodies[1].inv_mass;
    let (mut v2, mut force) = (vec![], vec![]);
    let mut last = w.bodies[1].vel.x;
    while w.bodies[1].vel.x > 8.0 {
        for _ in 0..30 {
            w.step(DT);
        }
        let v = w.bodies[1].vel.x;
        v2.push(((v + last) / 2.0).powi(2));
        force.push(mass * (last - v) / (30.0 * DT));
        last = v;
    }
    let drag = slope(&v2, &force);
    let n = v2.len() as f64;
    let rolling = (force.iter().sum::<f64>() - drag * v2.iter().sum::<f64>()) / n / (mass * 9.81);
    println!(
        "rolling resistance {rolling:.4} (Civic {civic_rolling:.4}); drag {drag:.3} N/(m/s)² (Civic {civic_drag:.3})"
    );
    assert!(
        (rolling - civic_rolling).abs() < 0.15 * civic_rolling,
        "rolling {rolling:.4}"
    );
    assert!(
        (drag - civic_drag).abs() < 0.1 * civic_drag,
        "drag {drag:.3}"
    );
}
