# Physics-simulator

Two small physics cores in Rust, compiled to WebAssembly, with an interactive
browser demo: a robot simulator in joint coordinates with motors, and a 3D
rigid-body engine for loose objects. Inspired by [MuJoCo](https://github.com/google-deepmind/mujoco);
not a port of it (see "What this is not").

## Run the demo

Live: https://aabhimittal.github.io/Physics-simulator/ (deployed from `web/` by
`.github/workflows/pages.yml` on every push to `main`). Or locally:

```sh
python3 -m http.server -d web      # then open http://localhost:8000
```

A walking person, a car on a circular road, and the two together: the car stopping
for the person crossing. Six robot scenes: a one-legged hopper that balances by foot placement, the same
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

`.github/workflows/ci.yml` runs on every pull request and push to `main`: `cargo fmt
--check`, the Rust and Python tests, the WebAssembly build, and, on pull requests, a
check that a change to simulation code also updates `web/sim.wasm`. The live site
(`.github/workflows/pages.yml`) builds `sim.wasm` from source, so it always matches `main`.

Scene ids live in one place, `src/scenes.rs`. The page's `SCENES` list must follow them;
it checks itself on load and logs `Scene list out of step with src/scenes.rs` to the
console if not. The walker's body parts, which the page draws as a person, come from
`robot::WALKER_PARTS`, and a test keeps that list in step with the walker's links.

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
  The springs are stiff, so each RK4 stage folds them into the mass matrix through each
  contact point's Jacobian (`Jᵀ(h·C + h²·K)J`): contact stays stable even at one step per
  frame, where the explicit version blew up.
- **Contact with world bodies** (`Robot::touch`): link corners inside a body, and box
  corners inside a link, are pushed apart by the same kind of spring, and the body gets the
  opposite impulse, so momentum passes both ways.
- **A moving base** is massless slide joints (x, y, and for the walker z) and the pitch
  hinge ahead of the body, so the hopper moves in a plane and the walker can also be shoved
  sideways.

Not there yet: tipping out of the plane (the walker can slide sideways but not fall
sideways), sensors, and loading models from a file.

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

The same training with and without the limit, seeds 0 to 3, each policy run
from 100 starts:

| | Peak leg speed | Peak leg power | Median speed | Runs that fell |
|---|---|---|---|---|
| No limit | 21 to 31 m/s | 8.7 to 13.9 kW | 4.1 to 5.6 m/s | 37 of 400 |
| Limit | 5.0 to 5.1 m/s | 560 W | 4.4 to 5.0 m/s | 1 of 400 |

Without the limit every seed drove the leg at speeds and powers no 11.5 kg robot
could reach. The limit barely changes how fast the hopper travels, because that
depends more on leg force than on motor power. The limited policies also fall
far less often, perhaps because a motor that weakens as it speeds up gives
smoother, more repeatable hops.

## Walking person

`robot::walker` is a 70 kg planar body: torso, thigh, shin and foot per leg, and an upper
arm and forearm per side, with motors about as strong as human muscle. It is driven by
SIMBICON (Yin, Loken and van de Panne, 2007): a phase cycle of spring-damper targets, the
stance hip holding the torso upright, and foot placement that steps further ahead the
faster and further the body is ahead of the stance foot. On top of that, to look less
like a robot:

- **Push-off**: while the trailing foot is still down the calf points the toe and drives
  the body on; it eases off when walking faster than the 1.6 m/s it aims for.
- **Stiff stance knee and ankle**, so the body walks tall instead of crouching (the
  original crouched to 48° of knee bend; now about 10°).
- **Reaching for the ground when the body says so**: the swing leg extends for landing
  after its swing time *or* once the stance leg is 0.25 rad behind the body. With time
  alone, a slightly fast body fell past its stance foot and stumbled every few steps.
- **Smooth targets** (blended over 0.1 s, as muscles ramp) and **swinging arms**, which
  are real links whose mass the balance has to carry.

It walks at 1.14 m/s, 1.9 steps a second, 0.6 m steps varying by under 10%, heel
first, within the range of adult walking, and survives shoves up to 250 N either way at
any point of the stride (36 of 36 in a sweep). (It walked a brisker 1.6 m/s until ground
contact became implicit; friction now wastes less of the push-off as foot slip.) It is
not fully human: it walks in a plane (no hip sway or pelvis twist), pushes off to about
-5° where people reach -16°, and keeps its stance knee straighter (10° against 15-20°). `tests/robot.rs` has an opt-in random-search tuner
(`cargo test --release --test robot tune_walk -- --ignored --nocapture`); it scores only
two shoves, so check what it finds against more before adopting it.

In the browser the links are drawn as a person (tapered limbs, head, hands, shoes,
clothes); the physics underneath is the same set of boxes.

## Car

`src/vehicle.rs` is a raycast vehicle, the usual game-engine design: a rigid body in
the `World` carried by four ray-cast spring-damper suspensions, with tyre forces from
Pacejka's Magic Formula capped by the friction circle. Rear-wheel drive through an open
differential from an engine with a torque curve (175 N·m at 4500 rpm, about 90 kW) and a
five-speed automatic gearbox with a rev limiter; front-biased brakes with ABS; Ackermann
steering; anti-roll bars; aero drag and rolling resistance. Each wheel's spin is stepped
implicitly, since the tyre makes it stiff. Each wheel's ray finds the nearest surface of
any body (ground, kerb, ramp, crate, log) and pushes back on it, so the car climbs ramps,
jumps off them, bumps over the speed hump and shoves loose objects.
`tests/vehicle.rs` checks it against textbook numbers: 100-0 km/h in 39.1 m against the
ideal v²/2μg = 39.3 m, 0-100 km/h in about 8 s with clean upshifts, sideways acceleration
v²/r on its circle with under 4° of body roll, a wheel standing on a block, and a ramp jump
that lands on its wheels.
In the browser it drives itself; arrow keys or WASD take over. It is drawn car-shaped
(an extruded side profile with a glass cabin, rounded edges), with headlights and tail
lights that light up with the simulated brake pedal, and the readout shows gear and rpm; the road has a
dashed centre line and, in the crossing scene, a zebra crossing. It shows load transfer,
body roll, understeer at full lock, and power oversteer (a steady drift) under full
throttle in a bend.

### Crossing

`scenes::crossing` puts the car and the walker together. They live in different
simulators, stepped together 16 times a frame with `Robot::touch` between them, so they
collide: a car that fails to stop throws the person along the road and loses the momentum
it gave them (tested against the same run with nobody in the road). The walker is upright
and planar, so it is shoved and dragged rather than thrown over the bonnet. The car's
autopilot is given the person's position and velocity, like a driver's eyes. Anyone on the road, or due on it within 3 s, makes it cap
its speed at √(2·a·d), the most from which braking at a = 6 m/s² still stops it in the
distance d left, aiming to stop with its nose 4 m short. It waits while they cross, then
drives on; a new person steps off the kerb each time. `tests/vehicle.rs` runs a minute
of it: several stops, never closer than 5.9 m centre to person.

Limits: a wheel is a single ray, so it meets a kerb's edge by snapping up onto it rather
than rolling over the corner (a shape cast is the upgrade); the gearbox is automatic only.

## Implicit joint damping

The robot integrator solves `(M + h·D + h²·K) q̈ = forces` each RK4 stage, where D and K
are each joint's own damping and stiffness (joint friction, joint limits, and a motor
past its free speed). Stiff dampers on light links such as a foot otherwise overshoot
every step; a knocked-down walker gained momentum from nothing and slid off at 100 m/s.
`tests/robot.rs` checks that it now comes to rest.

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
- Shapes are spheres, boxes, capsules and one ground plane; no convex hulls or meshes.
  Capsule against box samples spheres along the capsule, so a box edge can sink a little
  between samples. Joints are rods and ball joints only.
- The broad phase sorts and sweeps along x (1500 scattered bodies run faster than real
  time; checking every pair was 6.6 times slower). It slows when many bodies share an x
  range; a grid or BVH is the next step for a city.
- Stacks are slightly soft (a few mm of sink under an eight-row wall) and rely on
  sleeping to stay perfectly still.
