# Physics-simulator

Two small physics cores in Rust, compiled to WebAssembly, with an interactive
browser demo: a robot simulator in joint coordinates with motors, and a 3D
rigid-body engine for loose objects. Inspired by [MuJoCo](https://github.com/google-deepmind/mujoco);
not a port of it (see "What this is not").

## Run the demo

```sh
python3 -m http.server -d web      # then open http://localhost:8000
```

Six robot scenes: a one-legged hopper that balances by foot placement, the same
hopper run by a trained neural network, a cart-pole that swings its pole up and
balances it, a cart-pole that learns to balance while you watch, a
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
- **Torque-speed limit.** A motor with a `free_speed` loses force in a straight
  line as the joint speeds up in the direction it pushes, down to zero at that
  speed; braking keeps full force. This caps its power at `gear × free_speed / 4`.
  The hopper has it (200 W hip, 560 W leg); the cart-pole and arm motors do not.

`tests/robot.rs` checks the pendulum period, mass-matrix symmetry, energy
conservation, the cart-pole swing-up and recovery, and the arm's tracking error.

- **Ground contact** is a damped spring plus friction at every box corner below the floor.
- **A moving base** is three massless joints (x, y, pitch) ahead of the body, so the
  hopper moves in a plane.

Not there yet: contact between robots and other objects, motion out of the plane,
sensors, and loading models from a file.

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

### Training the hopper and watching the result

`python/train_hopper.py` trains PPO (Stable-Baselines3) on a task, keeps the best
checkpoint as `ppo_<task>.zip` and writes its network to `ppo_<task>.bin`:

```sh
pip install stable-baselines3
python3 python/train_hopper.py hop 1000000 0      # task, steps, seed; about 15 minutes on 4 cores
cp ppo_hop.bin src/hop_policy.bin                 # then rebuild web/sim.wasm as above
```

`src/hop_policy.bin` is compiled into the library. `env::hop_policy` runs the
network and the "Trained hopper" scene plays it back, one 10 s run after another.
`tests/env.rs` checks that the committed policy completes the full 10 s from 100
different starts.

The committed policy (seed 0) covers about 44 m in 10 s (4.4 m/s) at 2.4 hops per
second. Seeds 1 to 3 also learn to hop, at 46 to 50 m in 10 s. Seed 2's policy falls
on 1 of the 100 test starts, which the 10 evaluation episodes during training did
not show, so run the test before trusting a new policy.

Before the motors had a torque-speed limit, the same training produced a policy
that drove the leg at 21 m/s with 8.7 kW peaks, which no 11.5 kg robot could do.
The limit caps that at 5 m/s and 560 W. It lowers the top speed only a little
(5.6 m/s before), because running speed depends more on leg force than on motor power.

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
