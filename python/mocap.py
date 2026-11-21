"""Turns a CMU motion-capture walk into the walker's reference gait, src/mocap_walk.rs.

    python3 python/mocap.py 35.asf 35_02.amc > src/mocap_walk.rs

The files come from the CMU Graphics Lab Motion Capture Database
(http://mocap.cs.cmu.edu/subjects/35/), which may be copied, modified or
redistributed without permission. Subject 35, trial 2 is a plain walk.

The walker moves in a plane, so the capture is reduced to the sagittal plane:
forward kinematics gives every joint's 3D position, each limb is projected onto
the plane of the walking direction and vertical, and its angle there becomes the
walker's joint angle. One gait cycle (right heel strike to the next) is cut out,
and its two halves, right-leg stance then left-leg stance, are averaged into one
step, so the table describes "the stance leg" and "the swing leg".
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


def main():
    bones, children = read_asf(sys.argv[1])
    frames = [positions(bones, children, f) for f in read_amc(sys.argv[2])]
    # Walking direction from the root's travel; capture is y-up.
    travel = frames[-1]["root"] - frames[0]["root"]
    fwd = np.array([travel[0], 0.0, travel[2]])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    plane = lambda v: (v @ fwd, v @ up)  # noqa: E731

    def hanging(a, b):  # angle of limb a->b from straight down, positive forward
        x, y = plane(b - a)
        return np.arctan2(x, -y)

    def lying(a, b):  # angle of a->b above forward-horizontal
        x, y = plane(b - a)
        return np.arctan2(y, x)

    # The foot's rest direction (ankle to toe), with the shin vertical, is the walker's ankle 0.
    rest = positions(bones, children, {"root": [0.0] * 6})
    foot_rest = lying(rest["ltibia"], rest["lfoot"])

    def angles(p, s):  # side "l" or "r": torso pitch, hip, knee, ankle, shoulder, elbow
        x, y = plane(p["thorax"] - p["root"])
        torso = -np.arctan2(x, y)  # positive leaning back
        thigh = hanging(p[s + "hipjoint"], p[s + "femur"])
        shin = hanging(p[s + "femur"], p[s + "tibia"])
        foot = lying(p[s + "tibia"], p[s + "foot"]) - foot_rest
        upper = hanging(p[s + "clavicle"], p[s + "humerus"])
        fore = hanging(p[s + "humerus"], p[s + "radius"])
        return torso, thigh, shin - thigh, foot - shin, upper, fore - upper

    left = np.array([angles(p, "l") for p in frames])
    right = np.array([angles(p, "r") for p in frames])
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

    r_hits, l_hits = strikes("r"), strikes("l")
    r0 = r_hits[0]
    l0 = next(i for i in l_hits if i > r0)
    r1 = next(i for i in r_hits if i > l0)
    print(f"// right strikes {r_hits}, left strikes {l_hits}; using {r0}..{l0}..{r1}",
          file=sys.stderr)

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
    # own lean, and this subject carries the upper arms about 30 degrees forward throughout,
    # which drawn on the walker's simpler body looks like reaching rather than walking.
    torso -= torso.mean()
    arms = (stance[:, 4].mean() + swing[:, 4].mean()) / 2
    stance[:, 4] -= arms
    swing[:, 4] -= arms

    print("//! Reference gait from motion capture, written by python/mocap.py. Do not edit.")
    print("//! CMU Graphics Lab Motion Capture Database, subject 35, trial 2 (a walk).")
    print("//! One step, from the stance foot's heel strike to the swing foot's, in")
    print(f"//! {SAMPLES} samples. Angles in radians, in the walker's joint conventions.")
    print()
    print(f"/// How long one step took in the capture (s), and the walking speed (m/s).")
    print(f"pub const STEP_SECONDS: f64 = {step_seconds:.4f};")
    print(f"pub const SPEED: f64 = {speed:.4f};")
    print()
    print("/// Per sample: torso pitch about its mean; stance hip (thigh from vertical, in the")
    print("/// world), knee, ankle; swing hip (world), knee, ankle; stance-side shoulder (upper")
    print("/// arm from vertical, in the world), elbow; swing-side shoulder, elbow.")
    print(f"pub const STEP: [[f64; 11]; {SAMPLES}] = [")
    for k in range(SAMPLES):
        row = [torso[k], *stance[k, 1:4], *swing[k, 1:4], *stance[k, 4:6], *swing[k, 4:6]]
        print("    [" + ", ".join(f"{v:.4f}" for v in row) + "],")
    print("];")


if __name__ == "__main__":
    main()
