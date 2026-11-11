# Physics-simulator

A small 3D rigid-body physics engine in Rust, compiled to WebAssembly, with an
interactive browser demo. Inspired by [MuJoCo](https://github.com/google-deepmind/mujoco);
not a port of it (see "What this is not").

## Run the demo

```sh
python3 -m http.server -d web      # then open http://localhost:8000
```

Five scenes: a brick wall, dominoes, Newton's cradle, three double pendulums
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

## How it works

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
