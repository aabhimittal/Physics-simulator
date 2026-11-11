//! Narrowphase: turns a pair of overlapping shapes into contact points.

use crate::{Body, Quat, Shape, Vec3};

#[derive(Clone, Copy)]
pub struct Contact {
    pub a: usize,
    pub b: usize,
    /// Contact point on each body, in that body's frame.
    pub la: Vec3,
    pub lb: Vec3,
    /// Unit normal pointing from b to a.
    pub n: Vec3,
    /// Approach speed along `n` before the solver ran (negative = closing).
    pub vn_pre: f64,
    /// Normal impulse the solver applied this substep; 0 if it was not penetrating.
    pub normal_impulse: f64,
}

/// `pa` and `pb` are the world contact points; penetration depth is `(pb - pa)·n`.
fn push(bodies: &[Body], out: &mut Vec<Contact>, a: usize, b: usize, pa: Vec3, pb: Vec3, n: Vec3) {
    let (ba, bb) = (&bodies[a], &bodies[b]);
    out.push(Contact {
        a,
        b,
        n,
        la: ba.rot.inverse() * (pa - ba.pos),
        lb: bb.rot.inverse() * (pb - bb.pos),
        vn_pre: (ba.point_vel(pa - ba.pos) - bb.point_vel(pb - bb.pos)).dot(n),
        normal_impulse: 0.0,
    });
}

pub fn collide(bodies: &[Body], i: usize, j: usize, out: &mut Vec<Contact>) {
    let (a, b) = (&bodies[i], &bodies[j]);
    let mut on_plane = |p: Vec3| {
        let depth = b.pos.y - p.y;
        if depth > 0.0 {
            push(bodies, out, i, j, p, p + Vec3::Y * depth, Vec3::Y);
        }
    };
    match (a.shape, b.shape) {
        (Shape::Plane, Shape::Plane) => {}
        (Shape::Plane, _) | (Shape::Box(_), Shape::Sphere(_)) => collide(bodies, j, i, out),
        (Shape::Sphere(r), Shape::Plane) => on_plane(a.pos - Vec3::Y * r),
        (Shape::Box(h), Shape::Plane) => {
            for k in 0..8 {
                let sign =
                    Vec3::new((k & 1) as f64, (k >> 1 & 1) as f64, (k >> 2) as f64) * 2.0 - 1.0;
                on_plane(a.pos + a.rot * (h * sign));
            }
        }
        (Shape::Sphere(ra), Shape::Sphere(rb)) => {
            let d = a.pos - b.pos;
            let n = d.try_normalize().unwrap_or(Vec3::Y);
            if d.length() < ra + rb {
                push(bodies, out, i, j, a.pos - n * ra, b.pos + n * rb, n);
            }
        }
        (Shape::Sphere(r), Shape::Box(h)) => {
            let l = b.rot.inverse() * (a.pos - b.pos);
            let nearest = l.clamp(-h, h);
            let (nl, depth, surface) = if nearest == l {
                // Centre is inside the box: leave through the closest face.
                let gap = h - l.abs();
                let k = (0..3).min_by(|&x, &y| gap[x].total_cmp(&gap[y])).unwrap();
                let (mut nl, mut surface) = (Vec3::ZERO, l);
                nl[k] = l[k].signum();
                surface[k] = nl[k] * h[k];
                (nl, r + gap[k], surface)
            } else {
                let d = l - nearest;
                (d.normalize(), r - d.length(), nearest)
            };
            if depth > 0.0 {
                let n = b.rot * nl;
                push(bodies, out, i, j, a.pos - n * r, b.pos + b.rot * surface, n);
            }
        }
        (Shape::Box(ha), Shape::Box(hb)) => box_box(bodies, i, j, ha, hb, out),
    }
}

/// Sutherland–Hodgman: keeps the part of `poly` where `n·p <= d`.
fn clip(poly: &[Vec3], n: Vec3, d: f64) -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (dp, dq) = (n.dot(p) - d, n.dot(q) - d);
        if dp <= 0.0 {
            out.push(p);
        }
        if dp * dq < 0.0 {
            out.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    out
}

fn box_box(bodies: &[Body], ia: usize, ib: usize, ha: Vec3, hb: Vec3, out: &mut Vec<Contact>) {
    let (a, b) = (&bodies[ia], &bodies[ib]);
    let axes = |q: Quat| [q * Vec3::X, q * Vec3::Y, q * Vec3::Z];
    let (ax, bx) = (axes(a.rot), axes(b.rot));
    let t = a.pos - b.pos;
    let extent =
        |x: &[Vec3; 3], h: Vec3, l: Vec3| (0..3).map(|k| h[k] * x[k].dot(l).abs()).sum::<f64>();

    // Separating axis test over 3 + 3 face normals and 9 edge-pair cross products;
    // the axis of least overlap is the contact normal.
    let mut best = (f64::MAX, 0.0, Vec3::ZERO, 0);
    for id in 0..15 {
        let l = match id {
            0..3 => ax[id],
            3..6 => bx[id - 3],
            _ => ax[(id - 6) / 3].cross(bx[(id - 6) % 3]),
        };
        let Some(l) = l.try_normalize() else { continue }; // parallel edges
        let overlap = extent(&ax, ha, l) + extent(&bx, hb, l) - t.dot(l).abs();
        if overlap < 0.0 {
            return;
        }
        // Edge axes must win clearly, so resting boxes keep a stable face manifold.
        let score = if id < 6 {
            overlap
        } else {
            overlap * 1.05 + 1e-4
        };
        if score < best.0 {
            best = (score, overlap, l, id);
        }
    }
    let (_, overlap, l, id) = best;
    let n = if l.dot(t) < 0.0 { -l } else { l };

    if id >= 6 {
        // Edge against edge: one contact where the two closest edges cross.
        let (i, j) = ((id - 6) / 3, (id - 6) % 3);
        let (mut ea, mut eb) = (a.pos, b.pos);
        for k in 0..3 {
            if k != i {
                ea -= ax[k] * ha[k] * n.dot(ax[k]).signum();
            }
            if k != j {
                eb += bx[k] * hb[k] * n.dot(bx[k]).signum();
            }
        }
        let (r, d) = (ea - eb, ax[i].dot(bx[j]));
        let s = (d * bx[j].dot(r) - ax[i].dot(r)) / (1.0 - d * d);
        let pa = ea + ax[i] * s;
        push(bodies, out, ia, ib, pa, pa + n * overlap, n);
        return;
    }

    // Face contact: the reference box owns the axis. Clip the other box's facing
    // face against the reference face's four side planes; what is left below the
    // reference face is the contact patch.
    let (rf, hr, rx, inc, hi, ix, k, nr) = if id < 3 {
        (a, ha, &ax, b, hb, &bx, id, -n)
    } else {
        (b, hb, &bx, a, ha, &ax, id - 3, n)
    };
    let f = (0..3)
        .max_by(|&x, &y| ix[x].dot(nr).abs().total_cmp(&ix[y].dot(nr).abs()))
        .unwrap();
    let centre = inc.pos - ix[f] * hi[f] * ix[f].dot(nr).signum();
    let (u, v) = (
        ix[(f + 1) % 3] * hi[(f + 1) % 3],
        ix[(f + 2) % 3] * hi[(f + 2) % 3],
    );
    let mut poly = vec![
        centre + u + v,
        centre - u + v,
        centre - u - v,
        centre + u - v,
    ];
    for m in [(k + 1) % 3, (k + 2) % 3] {
        for side in [rx[m], -rx[m]] {
            // The 1 mm margin keeps corners that sit right on the boundary (aligned
            // stacks) from flickering in and out, which would shake the stack.
            poly = clip(&poly, side, side.dot(rf.pos) + hr[m] + 1e-3);
        }
    }
    let face = nr.dot(rf.pos) + hr[k];
    for p in poly {
        let depth = face - nr.dot(p);
        if depth > 0.0 {
            let on_ref = p + nr * depth;
            let (pa, pb) = if id < 3 { (on_ref, p) } else { (p, on_ref) };
            push(bodies, out, ia, ib, pa, pb, n);
        }
    }
}
