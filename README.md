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
--check`, `cargo clippy` with warnings as errors, the Rust and Python tests, the WebAssembly build, and, on pull requests, a
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
- **Static friction**: a corner that touches the floor is anchored there and pulled back by
  a stiff spring; only when that pull passes μ times the load does the anchor slide along
  (Coulomb friction). Friction as a damper alone, as it was, lets any steady sideways force
  creep a foot along: a person standing still slid 8 cm a second, and one stopped mid-stride
  did the splits.
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
let mut env = Env::new(Task::Balance);       // or Task::Hop, Task::Reach, Task::Imitate
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
arm and forearm per side, with motors about as strong as human muscle. It walks by
**imitating motion capture**, the DeepMimic approach (Peng et al., 2018) in miniature:

1. **Recorded walks at three speeds.** `python/mocap.py` reads subject 07 of the [CMU
   Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/) (which may be copied and
   redistributed without permission) walking slowly (trial 4, 0.93 m/s), normally (trial
   1, 1.36 m/s) and briskly (trial 12, 1.93 m/s). Forward kinematics gives every joint's 3D
   position; each limb is projected onto the side-on plane the walker moves in; one stride
   is cut out at the heel strikes and its left and right halves averaged into one step. The
   result is `src/mocap_walk.rs`: torso, hip, knee, ankle, shoulder and elbow angles for the
   stance and swing sides through one step of each walk.
2. **A tracking controller** (`robot::imitate`) aims every joint at the recorded angle for
   the pace it is asked for (`Robot::command`), blending between the two walks either side
   of it, with SIMBICON's balance rules on top (Yin, Loken and van de Panne, 2007): the
   stance hip holds the torso upright, and the swing foot lands further ahead the further
   and faster the body is ahead of the stance foot. Alone it cannot walk: joint springs pull
   only once a joint is already off target, so they trail the recording.
3. **A learned correction** (`robot::mimic_walk`): a network of two 64-unit layers adds up
   to ±0.3 rad to six of those targets (torso, swing hip, knee and ankle, stance knee and
   ankle). PPO trained it on `env::Task::Imitate`, rewarded equally for matching the
   recorded pose and the pace asked for. Each episode starts at a random point in the
   stride, or standing still, feet together or stopped mid-stride; about every three
   seconds the pace asked for changes (0.6 to 1.6 m/s, or a quarter of the time, stop);
   and about every two seconds the chest is shoved with up to 300 N. It sees and acts in
   stance/swing terms, so one network drives both legs.
4. **Standing still** is a separate, hand-written controller (`robot::stand`), as in most
   game characters and robots. Asked to stop, the walker keeps stepping until a footfall
   lands past the capture point (the centre of mass plus speed·√(leg/g), where a step must
   land for the body to come to rest), then holds its pose there, knees straight and feet
   flat; feet together, the ankles keep the centre of mass over the middle of the feet.
   Asked to go, it steps off with the rear foot. Because the stop happens inside the
   controller the policy trained with, the policy learned to arrive at a footfall the
   standing controller can hold; with a stop bolted on afterwards, it fell.

Measured over seconds 5 to 15 (normal pace unless stated):

| | Hand-tuned SIMBICON | One walk (before) | Three walks, pace control |
|---|---|---|---|
| Joint angles off the recording (rms) | 14.6° | 6.8° | 7.0° |
| Speed, cadence, step | 1.14 m/s, 1.9/s, 0.6 m | 1.18 m/s, 1.9/s, 0.61 m | 1.28 m/s, 1.8/s, 0.71 m |
| Asked 0.8 / 1.36 / 1.6 m/s, walks | | | 0.73 / 1.28 / 1.46 m/s |
| Stops and stands, then walks on | | | yes |
| 250 N shoves survived, both ways through a stride | 36 of 36 | 36 of 36 | 36 of 36 |

**Seeds.** Training is not deterministic in outcome, so three seeds were trained (2.5M steps
each, about 100 minutes with all three sharing 4 cores) and each checked against the whole
test suite: seed 0 passes every test and is the one committed; seed 1 fails one (its swing
knee peaks at 49°, the test asks for 50° or more); seed 2 fails two (8.9° off the recording
against 8°, and it falls while stopping). Before stops were part of training, all three fell
or failed to come to rest when stopping.

**Uneven ground.** The floor is a side profile, `Robot::terrain`: points joined by straight
lines (slopes, ramps; a step is two points a hair apart). A link corner below it is pushed
out to the nearest point of the profile, so off a slope square to the surface and off a
step's face sideways (a toe catches on a step rather than being lifted onto it); friction,
static friction's anchors included, works in the surface's own plane. The policy was trained
on flat ground and cannot see the terrain, yet it walks random courses of 8% slopes and 5 cm
steps (10 of 10), climbs 16%, and steps down 12 cm. It falls going down slopes of 12% or
more, and trips on steps up of 10 cm or more: its toe clears the ground by about 8 cm and it
does not know the step is there. So the crossing's pavement, 10 cm above the road, meets it
in dropped kerbs, 1.25 m ramps of 8% (about the steepest UK guidance allows), and the walking
scene's course keeps to ramps of 6% and steps of 4 cm. Letting the policy see the ground
ahead, and training it on rough terrain, is the upgrade for real kerbs and stairs.

It lands heel first, steps evenly, and once knocked down stops trying to walk (the muscles
let go). To retrain: `python3 python/train_hopper.py imitate 2500000 0`, then copy
`ppo_imitate.bin` over `src/walk_policy.bin` and run `cargo test --release`; trust the tests,
not the training log. Limits: it walks in a plane (no hip sway or pelvis twist) and cannot
turn; the brisk end is capped at 1.6 m/s (asked for the captured brisk walk's 1.9 m/s, it
fell within a second); it walks about 5 to 10% slower than asked; and standing still, it
withstands a nudge (about 120 N) but a hard shove topples it, where a person would step.

In the browser the links are drawn as a person (tapered limbs, head, hands, shoes,
clothes); the physics underneath is the same set of boxes.

## Car

`src/vehicle.rs` is a raycast vehicle, the usual game-engine design: a rigid body in
the `World` carried by four spring-damper suspensions, with tyre forces from Pacejka's Magic
Formula capped by the friction circle. Front-wheel drive through an open differential from
an engine with a torque curve (175 N·m at 4500 rpm, about 90 kW) and a five-speed automatic
gearbox with a rev limiter; front-biased brakes with ABS; Ackermann steering; anti-roll bars;
aero drag and rolling resistance. Each wheel's spin is stepped implicitly, since the tyre
makes it stiff.

- **Wheels are shape casts.** Each wheel sweeps a tyre-sized sphere down from its mount
  (`sphere_cast`: a ray against every shape grown by the tyre's radius, so boxes get rounded
  edges) and pushes back on whatever it finds: ground, kerb, ramp, crate, log. Meeting a kerb,
  the tyre touches its corner ahead of its lowest point and rolls up over it, rising about
  1 cm per cm of travel at first; the single ray it replaced found the kerb all at once and
  jolted the wheel up 10 cm in one step.
- **The gearbox shifts like an automatic**: its shift points follow the throttle (gently it
  changes up from 2000 rpm, flat out near the red line; off the throttle it changes down near
  idle), judged by road speed from the undriven wheels so wheelspin is not taken for revs.
  With fixed shift points it waited for 6200 rpm whatever the throttle, so the autopilot
  cruised at 50 km/h in first gear.

`tests/vehicle.rs` checks it against textbook numbers: 100-0 km/h in 39.0 m against the
ideal v²/2μg = 39.3 m, 0-100 km/h in 9.2 s with clean upshifts (about right for 90 kW; a
Civic 1.5T has about 130 kW and does 7 s), sideways acceleration v²/r on its circle with
under 4° of body roll, a wheel standing on a block, a 10 cm kerb, and a ramp jump that lands
on its wheels.

### Checked against a real car (comma2k19)

[comma2k19](https://github.com/commaai/comma2k19) (comma.ai, MIT licence) records a 2016
Honda Civic's CAN bus and a phone IMU over 33 hours of highway commuting. The car has the
Civic's 2.70 m wheelbase, so `tests/vehicle.rs` replays twelve one-minute Civic drives
(`tests/data/comma2k19.csv`, written by `python/comma2k19.py`, which reads only the few
arrays it needs out of the 10 GB chunks over HTTP range requests): the recorded
steering-wheel angle through the Civic's 15.38:1 ratio, and a cruise controller following
the recorded speed. It compares the simulated yaw rate with the phone gyro's.

The first replay found a real flaw. On the highway the Civic turns only about 0.65 times as
tightly as its wheels point (understeer), and the car turned 0.9 times. Its tyres' cornering
stiffness was exactly proportional to load, which makes understeer cancel out whatever the
weight distribution. Two changes from the textbook fixed it: tyres whose cornering stiffness
grows as load^0.5 (real tyres measure about 0.5 to 0.8), and the Civic's 61% of weight on
the front wheels. The exponent was chosen on 30 drives from chunk 3; the test uses drives
from chunk 5, recorded on other days:

| 30 held-out drives (chunk 5) | Before | After |
|---|---|---|
| Yaw gain, simulated minus recorded (mean of the 21 highway drives) | +0.25 | +0.03 |
| Yaw rate error against the gyro (rms, mean of all 30) | 0.0123 rad/s | 0.0087 rad/s |

Steering geometry alone (no tyre slip) is 0.0142 rad/s off. The committed test replays 12
of these drives and requires the highway yaw gain within 0.05 on average and 0.1 on each
drive; with the old tyres it fails, 0.23 off. The autopilot now adds the
steering an understeering car needs as speed rises (1 + 6·10⁻⁴·v²), as drivers learn to.
Slow, sharp turns read about 0.1 off, as the Civic Touring's variable-ratio steering is
quicker near lock than the fixed 15.38:1 used here.

**Road load, from the EPA.** comma2k19 cannot check the engine or brakes: in its raw CAN
logs the Civic's cruise control (openpilot) is engaged for most of each drive, the driver's
gas pedal reads 0 even while the car speeds up, and the moments a person coasts with
neither pedal are few, slow and mixed with the CVT's engine braking. What it cannot give, the
EPA's coast-down test does: every US car is coasted down from speed in neutral, and the
2016 Test Car List gives the Civic 4-door 1.5 L CVT a road load of 30.64 lbf - 0.0741 lbf/mph
· v + 0.01821 lbf/mph² · v² at 3250 lb. `tests/vehicle.rs` runs the same test on the car:
it coasts from 35 m/s and fits the force from its deceleration. Rolling resistance was 0.015
(textbook) against the Civic's 0.0094, which includes drivetrain friction; drag was within
4%. Both now come from the EPA figures (rolling 0.0094, Cd·A 0.675 m²), and the coast-down
reads 0.0091 and 0.391 N/(m/s)² against 0.0094 and 0.405. The engine itself is still
generic: a 90 kW, five-speed car, not the Civic's 130 kW turbo and CVT.

In the browser it drives itself; arrow keys or WASD take over. It is drawn car-shaped
(an extruded side profile with a glass cabin, rounded edges), with headlights and tail
lights that light up with the simulated brake pedal, and the readout shows gear and rpm; the road has a
dashed centre line and, in the crossing scene, a zebra crossing. It shows load transfer,
body roll, understeer that grows with speed, and the inside front wheel spinning under
full throttle in a tight bend (an open differential sends the torque where grip is least).

### Crossing

`scenes::crossing` puts the car and the walker together. They live in different
simulators, stepped together 16 times a frame with `Robot::touch` between them, so they
collide: a car that fails to stop throws the person along the road and loses the momentum
it gave them (tested against the same run with nobody in the road). The walker is upright
and planar, so it is shoved and dragged rather than thrown over the bonnet. The car's
autopilot is given the person's position and velocity, like a driver's eyes. Anyone on the road, or due on it within 3 s, makes it cap
its speed at √(2·a·d), the most from which braking at a = 6 m/s² still stops it in the
distance d left, aiming to stop with its nose 4 m short. As at a zebra crossing, it also
stops for someone waiting at the kerb, if it can do so gently (at 3 m/s²). It waits while
they cross, then drives on; a new person comes along each time. `tests/vehicle.rs` runs a
minute of it: several stops, never closer than 5.8 m centre to person. The driver never
speeds up toward someone in the way; it lifts off and lets the brakes do the rest.

The person judges the car as people do, through the walker's pace command
(`scenes::pace_for_crossing`): 2 m short of the road they decide, and if the car is under
3.5 s away and not visibly braking they slow to a stop at the kerb and wait; on the road, a
car under 2.5 s away and not braking makes them hurry (1.6 m/s). So most crossings are a
negotiation: the person waits, the car slows, the person goes.

The **Brakes fail** button (`scenes::crossing_brakes_failed`) restarts the scene with the
brake pedal doing nothing, the car placed 145° back round the road at 50 km/h, far enough
off that the person judges it safe to step out. The driver sees them and lifts off, the
person sees it coming and hurries, but coasting cannot stop it: it strikes them at about
45 km/h and slows as it passes them momentum. The timing is staged (starts from 130° to
160° back all lead to an impact); what follows is not. A person knocked down lies still for a moment before the next one steps off the kerb.

Limits: the tyre is a swept sphere, and a touch further to the side than its 11 cm half
tread counts as beside the tyre, not under it, so the wheel looks straight down instead (a
cylinder cast would do this exactly); the gearbox is automatic only.

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
