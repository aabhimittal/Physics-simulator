//! Demo scenes. Ids match the buttons in `web/index.html`.

use crate::{Body, Joint, Quat, Shape, Vec3, World};

/// Hangs a sphere on a rigid massless rod of length `len` from local point `lb`
/// of `parent`, swung by `angle` radians about z. Returns the new body's index.
pub fn hang(w: &mut World, parent: usize, lb: Vec3, radius: f64, len: f64, angle: f64) -> usize {
    let pivot = w.bodies[parent].pos + w.bodies[parent].rot * lb;
    let a = w.add(Body::new(
        Shape::Sphere(radius),
        1.0,
        pivot + Vec3::new(angle.sin(), -angle.cos(), 0.0) * len,
    ));
    w.joints.push(Joint {
        a,
        b: parent,
        la: Vec3::ZERO,
        lb,
        length: len,
        compliance: 0.0,
        damping: 0.0,
    });
    a
}

pub fn load(id: u32) -> World {
    let mut w = World::default();
    match id {
        // Brick wall to knock down.
        0 => {
            for row in 0..8 {
                let n = 8 - row;
                for i in 0..n {
                    let x = (i as f64 - (n - 1) as f64 / 2.0) * 0.82;
                    w.add(Body::new(
                        Shape::Box(Vec3::new(0.4, 0.2, 0.2)),
                        1.0,
                        Vec3::new(x, 0.2 + row as f64 * 0.4, 0.0),
                    ));
                }
            }
        }
        // Dominoes on an arc; the first one is already tipping.
        1 => {
            w.restitution = 0.0;
            let radius = 6.0;
            for i in 0..30 {
                let ang = i as f64 * 0.6 / radius - 1.5;
                let pos = Vec3::new(radius * ang.cos(), 0.5, radius * ang.sin());
                let mut d = Body::new(Shape::Box(Vec3::new(0.25, 0.5, 0.06)), 1.0, pos);
                d.rot = Quat::from_rotation_y(-ang); // thin axis along the arc
                if i == 0 {
                    d.omega = d.rot * Vec3::X * 2.0;
                }
                w.add(d);
            }
        }
        // Newton's cradle. The 2 mm gaps turn one five-body impact into a clean
        // chain of two-body impacts, which is also how a real cradle works.
        2 => {
            w.restitution = 1.0;
            w.friction = 0.0;
            for i in 0..5 {
                let x = (i as f64 - 2.0) * 0.502;
                hang(
                    &mut w,
                    0,
                    Vec3::new(x, 3.0, 0.0),
                    0.25,
                    2.0,
                    if i == 0 { -0.9 } else { 0.0 },
                );
            }
        }
        // Three double pendulums released 0.001 rad apart: chaos pulls them apart.
        3 => {
            for k in 0..3 {
                let angle = 2.2 + k as f64 * 1e-3;
                let upper = hang(
                    &mut w,
                    0,
                    Vec3::new(0.0, 3.4, (k as f64 - 1.0) * 0.5),
                    0.12,
                    1.0,
                    angle,
                );
                hang(&mut w, upper, Vec3::ZERO, 0.12, 1.0, angle);
            }
        }
        // Tennis racket theorem, in zero gravity: spin about the smallest and largest
        // inertia axes is stable, spin about the middle one flips over and over.
        _ => {
            w.gravity = Vec3::ZERO;
            for k in 0..3 {
                let mut b = Body::new(
                    Shape::Box(Vec3::new(0.5, 0.2, 0.05)),
                    1.0,
                    Vec3::new((k as f64 - 1.0) * 2.5, 2.5, 0.0),
                );
                b.omega = Vec3::splat(0.02);
                b.omega[k] = 6.0;
                w.add(b);
            }
        }
    }
    w
}
