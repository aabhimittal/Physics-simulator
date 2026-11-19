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
    use Shape::*;
    let (a, b) = (&bodies[i], &bodies[j]);
    match (a.shape, b.shape) {
        (Plane, Plane) => {}
        // Each pair is handled once, in one order.
        (Plane, _) | (Box(_), Sphere(_)) | (Box(_), Capsule(..)) | (Sphere(_), Capsule(..)) => {
            collide(bodies, j, i, out)
        }
        (Sphere(r), Plane) => on_plane(bodies, out, i, j, a.pos, r),
        (Box(h), Plane) => {
            for k in 0..8 {
                let sign =
                    Vec3::new((k & 1) as f64, (k >> 1 & 1) as f64, (k >> 2) as f64) * 2.0 - 1.0;
                on_plane(bodies, out, i, j, a.pos + a.rot * (h * sign), 0.0);
            }
        }
        // A capsule is a sphere swept along a segment: against anything, find the point of
        // the segment nearest the other shape and treat it as a sphere there.
        (Capsule(r, _), Plane) => {
            for end in segment(a) {
                on_plane(bodies, out, i, j, end, r);
            }
        }
        (Sphere(ra), Sphere(rb)) => spheres(bodies, out, i, a.pos, ra, j, b.pos, rb),
        (Sphere(r), Box(h)) => sphere_box(bodies, out, i, a.pos, r, j, h),
        (Capsule(ra, _), Sphere(rb)) => {
            let p = nearest_on_segment(segment(a), b.pos);
            spheres(bodies, out, i, p, ra, j, b.pos, rb);
        }
        (Capsule(ra, _), Capsule(rb, _)) => {
            let (p, q) = nearest_between_segments(segment(a), segment(b));
            spheres(bodies, out, i, p, ra, j, q, rb);
        }
        (Capsule(r, half), Box(h)) => {
            // shortcut: spheres sampled along the segment, so a box edge pressing in between
            // two samples sinks in by up to about a fifth of the radius; an exact
            // segment-box closest-point test is the upgrade if long thin capsules sink.
            let [p, q] = segment(a);
            let n = ((2.0 * half / r).ceil() as usize + 1).clamp(2, 9);
            for k in 0..n {
                let c = p.lerp(q, k as f64 / (n - 1) as f64);
                sphere_box(bodies, out, i, c, r, j, h);
            }
        }
        (Box(ha), Box(hb)) => box_box(bodies, i, j, ha, hb, out),
    }
}

/// The ends of a capsule's segment, in world coordinates.
pub fn segment(b: &Body) -> [Vec3; 2] {
    let Shape::Capsule(_, half) = b.shape else {
        return [b.pos; 2];
    };
    let axis = b.rot * Vec3::Y * half;
    [b.pos - axis, b.pos + axis]
}

/// A sphere of radius `r` centred at `c` (on body i) against the ground plane j.
fn on_plane(bodies: &[Body], out: &mut Vec<Contact>, i: usize, j: usize, c: Vec3, r: f64) {
    let p = c - Vec3::Y * r;
    let depth = bodies[j].pos.y - p.y;
    if depth > 0.0 {
        push(bodies, out, i, j, p, p + Vec3::Y * depth, Vec3::Y);
    }
}

/// Spheres at `pa` (on body a) and `pb` (on body b).
#[allow(clippy::too_many_arguments)]
fn spheres(
    bodies: &[Body],
    out: &mut Vec<Contact>,
    a: usize,
    pa: Vec3,
    ra: f64,
    b: usize,
    pb: Vec3,
    rb: f64,
) {
    let d = pa - pb;
    let n = d.try_normalize().unwrap_or(Vec3::Y);
    if d.length() < ra + rb {
        push(bodies, out, a, b, pa - n * ra, pb + n * rb, n);
    }
}

/// A sphere of radius `r` at `c` (on body i) against box j with half extents `h`.
fn sphere_box(
    bodies: &[Body],
    out: &mut Vec<Contact>,
    i: usize,
    c: Vec3,
    r: f64,
    j: usize,
    h: Vec3,
) {
    let b = &bodies[j];
    let l = b.rot.inverse() * (c - b.pos);
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
        push(bodies, out, i, j, c - n * r, b.pos + b.rot * surface, n);
    }
}

fn nearest_on_segment([p, q]: [Vec3; 2], x: Vec3) -> Vec3 {
    let d = q - p;
    let t = ((x - p).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p + d * t
}

/// The closest pair of points between segments p0-p1 and q0-q1 (Ericson, "Real-Time
/// Collision Detection", 5.1.9).
pub fn nearest_between_segments([p0, p1]: [Vec3; 2], [q0, q1]: [Vec3; 2]) -> (Vec3, Vec3) {
    let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
    let (a, e, f) = (d1.length_squared(), d2.length_squared(), d2.dot(r));
    let (c, b) = (d1.dot(r), d1.dot(d2));
    let denom = a * e - b * b;
    let mut s = if denom > 1e-12 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = if e > 1e-12 { (b * s + f) / e } else { 0.0 };
    if t < 0.0 {
        t = 0.0;
        s = if a > 1e-12 {
            (-c / a).clamp(0.0, 1.0)
        } else {
            0.0
        };
    } else if t > 1.0 {
        t = 1.0;
        s = if a > 1e-12 {
            ((b - c) / a).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    (p0 + d1 * s, q0 + d2 * t)
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

/// Where a ray first meets a body: distance along the (unit) direction, the surface normal
/// there, and which body.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub distance: f64,
    pub normal: Vec3,
    pub body: usize,
}

/// The nearest body surface along the ray from `from` in unit direction `dir`, within `max`,
/// ignoring body `skip`. A ray starting inside a body does not hit it.
pub fn raycast(bodies: &[Body], from: Vec3, dir: Vec3, max: f64, skip: usize) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for (i, b) in bodies.iter().enumerate() {
        if i == skip {
            continue;
        }
        let hit = match b.shape {
            // The plane is the half space below the body's height.
            Shape::Plane => {
                let t = (from.y - b.pos.y) / -dir.y;
                (dir.y < 0.0 && t >= 0.0).then_some((t, Vec3::Y))
            }
            Shape::Sphere(r) => ray_sphere(from - b.pos, dir, r),
            Shape::Capsule(r, half) => ray_capsule(from, dir, segment(b), r, half),
            Shape::Box(h) => {
                // Slab test in the box's own frame.
                let (o, d) = (b.rot.inverse() * (from - b.pos), b.rot.inverse() * dir);
                let (mut near, mut far, mut axis) = (f64::NEG_INFINITY, f64::INFINITY, 0);
                for k in 0..3 {
                    let (a, b2) = ((-h[k] - o[k]) / d[k], (h[k] - o[k]) / d[k]);
                    let (lo, hi) = (a.min(b2), a.max(b2));
                    if lo > near {
                        (near, axis) = (lo, k);
                    }
                    far = far.min(hi);
                }
                (near >= 0.0 && near <= far).then(|| {
                    let mut n = Vec3::ZERO;
                    n[axis] = -d[axis].signum();
                    (near, b.rot * n)
                })
            }
        };
        if let Some((t, normal)) = hit
            && t <= max
            && best.is_none_or(|h| t < h.distance)
        {
            best = Some(Hit {
                distance: t,
                normal,
                body: i,
            });
        }
    }
    best
}

/// Ray from `o` (relative to the sphere's centre) along unit `d` against a sphere of radius `r`.
fn ray_sphere(o: Vec3, d: Vec3, r: f64) -> Option<(f64, Vec3)> {
    let b = o.dot(d);
    let c = o.length_squared() - r * r;
    let disc = b * b - c;
    if c < 0.0 || disc < 0.0 {
        return None; // starts inside, or misses
    }
    let t = -b - disc.sqrt();
    (t >= 0.0).then(|| (t, (o + d * t) / r))
}

/// Ray against a capsule: its cylinder, then its two end spheres; the nearest hit wins.
fn ray_capsule(from: Vec3, d: Vec3, [p, q]: [Vec3; 2], r: f64, half: f64) -> Option<(f64, Vec3)> {
    let axis = (q - p) / (2.0 * half).max(1e-12);
    let mut best = [p, q]
        .into_iter()
        .filter_map(|c| ray_sphere(from - c, d, r))
        .min_by(|x, y| x.0.total_cmp(&y.0));
    // The cylinder: solve |(o + t d) without its axis part| = r.
    let o = from - p;
    let (dp, op) = (d - axis * d.dot(axis), o - axis * o.dot(axis));
    let (a, b, c) = (dp.length_squared(), dp.dot(op), op.length_squared() - r * r);
    if a > 1e-12 && c > 0.0 && b * b - a * c >= 0.0 {
        let t = (-b - (b * b - a * c).sqrt()) / a;
        let along = (o + d * t).dot(axis);
        if t >= 0.0 && (0.0..=2.0 * half).contains(&along) && best.is_none_or(|h| t < h.0) {
            best = Some((t, (op + dp * t) / r));
        }
    }
    best
}

/// How deep world point `p` is inside body `b`, and the way out (unit, from the body).
/// `None` if outside, and for the ground plane, which robots handle themselves.
pub fn penetration(b: &Body, p: Vec3) -> Option<(f64, Vec3)> {
    let round = |nearest: Vec3, r: f64| {
        let d = p - nearest;
        let len = d.length();
        (len < r).then(|| (r - len, d.try_normalize().unwrap_or(Vec3::Y)))
    };
    match b.shape {
        Shape::Plane => None,
        Shape::Sphere(r) => round(b.pos, r),
        Shape::Capsule(r, _) => round(nearest_on_segment(segment(b), p), r),
        Shape::Box(h) => inside_box(b.pos, b.rot, h, p),
    }
}

/// Depth and way out of a point inside a box centred at `c`, turned by `rot`.
pub fn inside_box(c: Vec3, rot: Quat, h: Vec3, p: Vec3) -> Option<(f64, Vec3)> {
    let l = rot.inverse() * (p - c);
    let gap = h - l.abs();
    if gap.min_element() <= 0.0 {
        return None;
    }
    let k = (0..3).min_by(|&x, &y| gap[x].total_cmp(&gap[y])).unwrap();
    let mut n = Vec3::ZERO;
    n[k] = l[k].signum();
    Some((gap[k], rot * n))
}
