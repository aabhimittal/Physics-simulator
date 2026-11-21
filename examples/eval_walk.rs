use physics_simulator::robot::{self, Robot};
fn run(name: &str, mut r: Robot) {
    let (mut err, mut n, mut fell) = (0.0, 0.0, None);
    let (mut x0, mut s0) = (0.0, 0.0);
    for f in 0..1200 {
        r.step(1.0 / 60.0, 16);
        if r.q[1] < 0.7 || r.q[3].abs() > 0.6 {
            fell = Some(r.time);
            break;
        }
        if f == 300 {
            x0 = r.q[0];
            s0 = r.memory[2];
        }
        if f > 300 {
            err += robot::imitation_error(&r);
            n += 1.0;
        }
    }
    println!(
        "{name}: fell {fell:?}, speed {:.2} m/s, cadence {:.2}/s, pose error {:.1} deg rms",
        (r.q[0] - x0) / 15.0,
        (r.memory[2] - s0) / 15.0,
        (err / n).sqrt().to_degrees()
    );
}
fn main() {
    run("SIMBICON (current)", robot::walker());
    let mut r = robot::walker();
    r.controller = Some(robot::imitate);
    run("tracking only", r);
    let mut r = robot::walker();
    r.controller = Some(robot::mimic_walk);
    run("tracking + policy", r);
}
