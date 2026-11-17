//! Checks the car against textbook vehicle dynamics.

use glam::DVec3 as Vec3;
use physics_simulator::World;
use physics_simulator::scenes::{self, CAR_SCENE};

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
    for frame in 0..(40.0 / DT) as usize {
        w.step(DT);
        let (b, car) = (&w.bodies[w.cars[0].body], &w.cars[0]);
        let off = (b.pos.x.hypot(b.pos.z) - 30.0).abs();
        assert!(
            off < 1.0,
            "{off:.2} m off the circle at t = {:.1}",
            car.time
        );
        // At steady speed the sideways acceleration is v² / r.
        if frame % 960 == 420 {
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
        }
    }
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
