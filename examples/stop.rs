use physics_simulator::robot;
fn main() {
    let mut r = robot::walker();
    let dt = 1.0 / 60.0;
    for frame in 0..480 {
        let wanted = if (180..480).contains(&frame) {
            0.0
        } else {
            robot::WALKING_SPEED
        };
        r.command += (wanted - r.command).clamp(-0.8 * dt, 0.8 * dt);
        r.step(dt, 16);
        if frame >= 240 && frame % 15 == 0 {
            let f = r.frames(&r.q);
            println!(
                "t {:.2} cmd {:.2} stand {} x {:.2} vx {:+.2} y {:.2} feet {:.2} {:.2} h {:.3} {:.3}",
                r.time,
                r.command,
                r.memory[4],
                r.q[0],
                r.qd[0],
                r.q[1],
                f[6].0.x,
                f[9].0.x,
                f[6].0.y,
                f[9].0.y
            );
        }
        if r.q[1] < 0.7 {
            println!("fell {:.2}", r.time);
            break;
        }
    }
}
