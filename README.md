# Physics-simulator

Two small physics cores in Rust, compiled to WebAssembly, with an interactive
browser demo: a robot simulator in joint coordinates with motors, and a 3D
rigid-body engine for loose objects. Inspired by [MuJoCo](https://github.com/google-deepmind/mujoco);
not a port of it (see "What this is not").

## Run the demo

```sh
python3 -m http.server -d web      # then open http://localhost:8000
```

Four robot scenes: a one-legged hopper that balances by foot placement, a cart-pole that swings its pole up and balances it, a
four-joint arm whose tip follows a moving target, and a motorless double
pendulum. Six rigid-body scenes: a block tower, a brick wall, dominoes, Newton's cradle, three double pendulums
(chaos), and the tennis racket theorem in zero gravity. Drag any object, drag
empty space to orbit, scroll to zoom, double-click to fire a ball.

## Build and test

```sh
cargo test --release
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/physics_simulator.wasm web/sim.wasm
```

`web/sim.wasm` is committed so the demo runs without a toolchain. Rebuild it
after changing anything under `src/`.

## How the robot simulator works

`src/robot.rs`. A robot is a tree of links, each joined to its parent by a
hinge or a slide. The state is one number per joint, so joints cannot come apart.

- **Inverse dynamics** (recursive Newton-Euler) returns the joint forces that
  produce a given motion. Called with different inputs it also gives the mass
  matrix, the gravity and Coriolis forces, and the joint effect of a force at any point.
- **Forward dynamics** solves `M·q̈ = motor + friction − bias`, then integrates with RK4.
- **Motors** take a command in [-1, 1] scaled by the joint's `gear`. A robot's
  `controller` function sets the commands every substep (480 Hz).

`tests/robot.rs` checks the pendulum period, mass-matrix symmetry, energy
conservation, the cart-pole swing-up and recovery, and the arm's tracking error.

- **Ground contact** is a damped spring plus friction at every box corner below the floor.
- **A moving base** is three massless joints (x, y, pitch) ahead of the body, so the
  hopper moves in a plane.

Not there yet: contact between robots and other objects, motion out of the plane,
joint limits, sensors, and loading models from a file.

## Reinforcement-learning interface

`src/env.rs` wraps the robots in the usual contract:

```rust
let mut env = Env::new(Task::Balance);       // or Task::Hop, Task::Reach
let mut obs = env.reset(seed);               // same seed, same start
loop {
    let step = env.step(&action);            // one number in [-1, 1] per motor
    // step.obs, step.reward, step.terminated (failed), step.truncated (10 s limit)
}
```

`Learner` is a small hill-climbing trainer over a linear policy. It learns to
balance the cart-pole in a few dozen episodes; the "Learning to balance" scene
runs it live. `tests/env.rs` checks sizes, seeding, termination, the time limit,
bad actions, and that learning succeeds for five seeds.

### From Python

`python/physics_simulator.py` calls the Rust library through its C interface
(`src/capi.rs`) using only the standard library:

```sh
cargo build --release
PYTHONPATH=python python3 -c "
from physics_simulator import Env
env = Env('hop')                      # or 'balance', 'reach'
obs = env.reset(seed=1)
obs, reward, terminated, truncated = env.step([0.0, 0.0])
"
python3 -m unittest discover python   # tests
```

With `gymnasium` and `numpy` installed, `GymEnv('hop')` is a standard
`gymnasium.Env` and passes Gymnasium's environment checker, so it can be handed
to RL libraries that accept Gymnasium environments.

## How the rigid-body engine works

Each 60 Hz step is split into 20 substeps (`World::step` in `src/lib.rs`):

1. **Integrate.** Gravity, then free rotation with RK4 on Euler's equations in the body frame.
2. **Find contacts.** Sphere, box and ground-plane pairs. Box against box uses the
   separating axis test plus face clipping (`src/collide.rs`).
3. **Solve joints** on positions. Rod joints push along the rod's direction at the
   start of the substep (SHAKE), which keeps pendulum energy.
4. **Solve contacts** on positions (XPBD): all normals at once, then Coulomb friction.
5. **Derive velocities** from how far each body actually moved.
6. **Restitution**, and bodies that stay still for half a second fall asleep.

`tests/physics.rs` checks the result against known answers: free fall, the
pendulum period, sliding distance under friction, energy and angular momentum
of a free spinning body, momentum transfer in Newton's cradle.

## What this is not

- Not MuJoCo. No generalized coordinates, actuators, tendons, sensors or MJCF
  model files, and no convex contact solver.
- Shapes are spheres, boxes and one ground plane. Joints are rods and ball joints only.
- Collision search checks every pair of bodies, so it suits a few hundred bodies.
- Stacks are slightly soft (a few mm of sink under an eight-row wall) and rely on
  sleeping to stay perfectly still.
