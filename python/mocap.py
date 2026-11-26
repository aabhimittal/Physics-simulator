"""Turns CMU motion-capture walks into the walker's reference gaits, src/mocap_walk.rs.

    python3 python/mocap.py 07.asf 07_04.amc 07_01.amc 07_12.amc > src/mocap_walk.rs
    python3 python/mocap.py --getup 140.asf 140_01.amc 140_02.amc 140_08.amc 140_09.amc > src/mocap_getup.rs

The files come from the CMU Graphics Lab Motion Capture Database
(http://mocap.cs.cmu.edu/subjects/07/), which may be copied, modified or
redistributed without permission. Subject 07 walked slowly (trial 4), normally (trial 1)
and briskly (trial 12): one person at three speeds, which the walker blends between.

The walker moves in a plane, so the capture is reduced to the sagittal plane:
forward kinematics gives every joint's 3D position, each limb is projected onto
the plane of the walking direction and vertical, and its angle there becomes the
walker's joint angle. One gait cycle (right heel strike to the next) is cut out,
and its two halves, right-leg stance then left-leg stance, are averaged into one
step, so each table describes "the stance leg" and "the swing leg".
"""

import sys

import numpy as np

SAMPLES = 50  # per step


def rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i = "xyz".index(axis)
    m = np.eye(3)
    j, k = (i + 1) % 3, (i + 2) % 3
    m[j, j], m[j, k], m[k, j], m[k, k] = c, -s, s, c
    return m


def euler(x, y, z):
    """ASF "axis ... XYZ": rotate about x, then y, then z (applied to column vectors)."""
    return rot("z", z) @ rot("y", y) @ rot("x", x)


def read_asf(path):
    bones, children, section, bone = {}, {}, None, None
    for line in open(path):
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0].startswith(":"):
            section = words[0]
            continue
        if section == ":bonedata":
            if words[0] == "begin":
                bone = {"dof": []}
            elif words[0] == "end":
                bones[bone["name"]] = bone
            elif words[0] == "name":
                bone["name"] = words[1]
            elif words[0] == "direction":
                bone["direction"] = np.array([float(w) for w in words[1:4]])
            elif words[0] == "length":
                bone["length"] = float(words[1])
            elif words[0] == "axis":
                bone["axis"] = euler(*[float(w) for w in words[1:4]])
            elif words[0] == "dof":
                bone["dof"] = words[1:]
        elif section == ":hierarchy" and words[0] not in ("begin", "end"):
            children[words[0]] = words[1:]
    return bones, children


def read_amc(path):
    frames, frame = [], None
    for line in open(path):
        words = line.split()
        if not words or words[0].startswith(("#", ":")):
            continue
        if words[0].isdigit() and len(words) == 1:
            frame = {}
            frames.append(frame)
        else:
            frame[words[0]] = [float(w) for w in words[1:]]
    return frames


def positions(bones, children, frame):
    """World position of the far end of every bone, plus the root."""
    out = {}
    root = frame["root"]
    out["root"] = np.array(root[:3])
    stack = [(c, euler(*root[3:6]), out["root"]) for c in children["root"]]
    while stack:
        name, parent, start = stack.pop()
        b = bones[name]
        motion = np.eye(3)
        for dof, value in zip(b["dof"], frame.get(name, [])):
            motion = motion @ rot(dof[1], value)  # dof order, as in the ASF/AMC spec
        world = parent @ b["axis"] @ motion @ b["axis"].T
        end = start + world @ b["direction"] * b["length"]
        out[name] = end
        stack += [(c, world, end) for c in children.get(name, [])]
    return out


def angles(p, s, plane, foot_rest):
    """Side "l" or "r" of pose `p`, in the walker's conventions in `plane` (a function giving a
    vector's forward and up parts): torso pitch (positive leaning back), thigh from straight
    down in the world (positive forward), knee, ankle, upper arm from straight down in the
    world, elbow."""

    def hanging(a, b):  # angle of limb a->b from straight down, positive forward
        x, y = plane(b - a)
        return np.arctan2(x, -y)

    def lying(a, b):  # angle of a->b above forward-horizontal
        x, y = plane(b - a)
        return np.arctan2(y, x)

    x, y = plane(p["thorax"] - p["root"])
    torso = -np.arctan2(x, y)
    thigh = hanging(p[s + "hipjoint"], p[s + "femur"])
    shin = hanging(p[s + "femur"], p[s + "tibia"])
    foot = lying(p[s + "tibia"], p[s + "foot"]) - foot_rest
    upper = hanging(p[s + "clavicle"], p[s + "humerus"])
    fore = hanging(p[s + "humerus"], p[s + "radius"])
    return torso, thigh, shin - thigh, foot - shin, upper, fore - upper


def rest_foot(bones, children, plane):
    """The foot's rest direction (ankle to toe) with the shin vertical: the walker's ankle 0."""
    rest = positions(bones, children, {"root": [0.0] * 6})
    x, y = plane(rest["lfoot"] - rest["ltibia"])
    return np.arctan2(y, x)


def clip(bones, children, amc):
    """One averaged step of the walk in `amc`: (speed m/s, step seconds, rows of 11 angles)."""
    frames = [positions(bones, children, f) for f in read_amc(amc)]
    # Walking direction from the root's travel; capture is y-up.
    travel = frames[-1]["root"] - frames[0]["root"]
    fwd = np.array([travel[0], 0.0, travel[2]])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    plane = lambda v: (v @ fwd, v @ up)  # noqa: E731

    foot_rest = rest_foot(bones, children, plane)
    left = np.array([angles(p, "l", plane, foot_rest) for p in frames])
    right = np.array([angles(p, "r", plane, foot_rest) for p in frames])

    # Heel strikes: where each ankle, swinging forward, comes almost to a stop (a fifth of its
    # fastest), which holds even where the capture's floor is not quite level.
    def strikes(s):
        x = np.array([p[s + "tibia"] @ fwd for p in frames])
        v = np.gradient(x)
        slow = 0.2 * v.max()
        hits = []
        for i in range(1, len(v)):
            if v[i] < slow <= v[i - 1] and (not hits or i - hits[-1] > 30):  # one per step
                hits.append(i)
        return hits

    # One stride, right heel strike to the next (at least half a second apart), and the left
    # heel strike nearest its middle.
    r_hits, l_hits = strikes("r"), strikes("l")
    r0 = r_hits[0]
    r1 = next(i for i in r_hits if i - r0 > 60)
    l0 = min((i for i in l_hits if r0 < i < r1), key=lambda i: abs(i - (r0 + r1) / 2))
    print(f"// {amc}: right strikes {r_hits}, left {l_hits}; using {r0}..{l0}..{r1}", file=sys.stderr)

    def resample(series, a, b):
        t = np.linspace(a, b, SAMPLES, endpoint=False)
        return np.array([np.interp(t, np.arange(len(series)), series[:, k])
                         for k in range(series.shape[1])]).T

    # Step 1: right foot is stance, left swings; step 2: the other way round.
    stance = (resample(right, r0, l0) + resample(left, l0, r1)) / 2
    swing = (resample(left, r0, l0) + resample(right, l0, r1)) / 2
    step_seconds = (r1 - r0) / 2 / 120.0  # captured at 120 Hz
    scale = 0.0254 / 0.45  # ASF length units to metres
    speed = np.linalg.norm(travel) * scale / ((len(frames) - 1) / 120.0)
    torso = (stance[:, 0] + swing[:, 0]) / 2
    # Keep only the swing of the torso and upper arms about their means: the walker holds its
    # own lean and lets its arms hang, as captured arms carried forward look like reaching.
    torso -= torso.mean()
    arms = (stance[:, 4].mean() + swing[:, 4].mean()) / 2
    stance[:, 4] -= arms
    swing[:, 4] -= arms
    rows = [[torso[k], *stance[k, 1:4], *swing[k, 1:4], *stance[k, 4:6], *swing[k, 4:6]] for k in range(SAMPLES)]
    return speed, step_seconds, rows


def getup(bones, children, amc):
    """Every fourth frame (30 Hz) of a get-up in `amc`, up to where the hips are highest (the
    person is still bent there, in a ready crouch, not standing tall):
    torso pitch, then each leg's thigh, knee and ankle, then each arm's upper arm and elbow,
    both sides kept (a get-up is not symmetric). The plane is the one the person faces once
    up (these get-ups turn less than 20 degrees on the way)."""
    frames = [positions(bones, children, f) for f in read_amc(amc)]
    # Up is where the hips are highest; after it the person settles into a ready crouch.
    frames = frames[: 1 + int(np.argmax([p["root"][1] for p in frames]))]
    up = np.array([0.0, 1.0, 0.0])
    end = frames[-1]
    right = end["rhipjoint"] - end["lhipjoint"]
    fwd = np.cross(up, right)
    fwd[1] = 0.0
    fwd /= np.linalg.norm(fwd)
    if (end["lfoot"] - end["ltibia"]) @ fwd < 0:  # toes point the way the person faces
        fwd = -fwd
    plane = lambda v: (v @ fwd, v @ up)  # noqa: E731
    foot_rest = rest_foot(bones, children, plane)
    rows = []
    for p in frames[::4]:
        torso, *left = angles(p, "l", plane, foot_rest)
        _, *right_ = angles(p, "r", plane, foot_rest)
        rows.append([torso, *left[:3], *right_[:3], *left[3:], *right_[3:]])
    rows = np.array(rows)
    # Torso pitch can pass through ±π (face down to face up); keep it continuous.
    rows[:, 0] = np.unwrap(rows[:, 0])
    print(f"// {amc}: {len(rows)} poses, from torso {rows[0, 0]:.2f} to {rows[-1, 0]:.2f}",
          file=sys.stderr)
    assert abs(rows[0, 0]) > 1.0, "does not start lying down"
    return rows


def main_getup(asf, amcs):
    bones, children = read_asf(asf)
    rows = np.concatenate([getup(bones, children, amc) for amc in amcs])
    names = " ".join(a.rsplit("/", 1)[-1].removesuffix(".amc") for a in amcs)
    print("//! Get-up poses from motion capture, written by python/mocap.py --getup. Do not edit.")
    print(f"//! CMU Graphics Lab Motion Capture Database, trials {names}: one person getting up")
    print("//! from lying face down and on their back, at 30 Hz, until the hips are highest (still")
    print("//! in a ready crouch). Angles in radians, in the walker's joint conventions.")
    print("#![allow(clippy::approx_constant)] // measured angles, some near π fractions by chance")
    print()
    print("/// Per pose: torso pitch (positive leaning back); each leg's thigh from straight down")
    print("/// in the world, knee and ankle; each arm's upper arm from straight down in the world")
    print("/// and elbow.")
    print(f"pub static POSES: [[f64; 11]; {len(rows)}] = [")
    for row in rows:
        print("    [" + ", ".join(f"{v:.4f}" for v in row) + "],")
    print("];")


def main():
    if sys.argv[1] == "--getup":
        return main_getup(sys.argv[2], sys.argv[3:])
    bones, children = read_asf(sys.argv[1])
    clips = sorted(clip(bones, children, amc) for amc in sys.argv[2:])
    names = " ".join(a.rsplit("/", 1)[-1].removesuffix(".amc") for a in sys.argv[2:])
    print("//! Reference gaits from motion capture, written by python/mocap.py. Do not edit.")
    print(f"//! CMU Graphics Lab Motion Capture Database, trials {names}: one person walking")
    print(f"//! at {len(clips)} speeds. Each is one step, from the stance foot's heel strike to the swing")
    print(f"//! foot's, in {SAMPLES} samples. Angles in radians, in the walker's joint conventions.")
    print("#![allow(clippy::approx_constant)] // measured angles, some near 1/π or log10(2) by chance")
    print()
    print("/// One captured walk, averaged into a single step.")
    print("pub struct Clip {")
    print("    /// Walking speed (m/s) and how long one step took (s).")
    print("    pub speed: f64,")
    print("    pub step_seconds: f64,")
    print("    /// Per sample: torso pitch about its mean; stance hip (thigh from vertical, in the")
    print("    /// world), knee, ankle; swing hip (world), knee, ankle; stance-side shoulder (upper")
    print("    /// arm from vertical, in the world), elbow; swing-side shoulder, elbow.")
    print(f"    pub step: [[f64; 11]; {SAMPLES}],")
    print("}")
    print()
    print("/// Slowest first.")
    print(f"pub const CLIPS: [Clip; {len(clips)}] = [")
    for speed, seconds, rows in clips:
        print(f"    Clip {{\n        speed: {speed:.4f},\n        step_seconds: {seconds:.4f},\n        step: [")
        for row in rows:
            print("            [" + ", ".join(f"{v:.4f}" for v in row) + "],")
        print("        ],\n    },")
    print("];")


if __name__ == "__main__":
    main()
