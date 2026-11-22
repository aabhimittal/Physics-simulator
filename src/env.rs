//! Reinforcement-learning environments over the robots: the usual
//! `reset` / `step` / observation / reward contract (as in Gymnasium).
//!
//! An action is one number in [-1, 1] per motor, in joint order.

use crate::Vec3;
use crate::robot::{self, ARM_TIP, Robot};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Task {
    /// Cart-pole: keep the pole upright and the cart on the rail.
    /// Observation `[cart x, cart speed, pole angle, pole rate]`; reward 1 per step alive.
    Balance,
    /// Hopper: move in +x without falling.
    /// Observation `[height, pitch, hip, leg, then the five joint rates]`;
    /// reward = forward speed + 1 alive - a small penalty on motor effort.
    Hop,
    /// Arm: bring the tip to a random target.
    /// Observation `[four joint angles, four rates, target - tip]`; reward = -distance.
    Reach,
    /// Walker: walk like the motion capture (`robot::imitate`), the policy correcting the
    /// controller's joint targets. Observation and action are in stance/swing order (see
    /// `imitation_obs`); reward = how closely the pose matches the capture, plus walking at
    /// the speed asked for, which changes now and then between 0.6 and 2 m/s; a quarter of
    /// episodes start standing still.
    Imitate,
}

/// Corrections the imitation policy makes: torso lean, swing hip, knee and ankle, stance
/// knee and ankle; each up to `IMITATE_REACH` radians either way.
pub const IMITATE_ACTIONS: usize = 6;
pub const IMITATE_REACH: f64 = 0.3;
/// The paces (m/s) the walking policy is trained for.
pub const WALKING_PACE: (f64, f64) = (robot::SLOWEST_WALK, 2.0);

/// Result of one `Env::step`.
pub struct Step {
    pub obs: Vec<f64>,
    pub reward: f64,
    /// The task failed (pole or robot fell); the episode is over.
    pub terminated: bool,
    /// The time limit was reached; the episode is cut off, not failed.
    pub truncated: bool,
}

/// Small deterministic random number generator (xorshift64*), so a seed fixes an episode.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        bits as f64 / (1u64 << 52) as f64 - 1.0
    }
}

pub struct Env {
    pub robot: Robot,
    pub task: Task,
    steps: u32,
    /// Random shoves while imitating: the generator, and the shove's steps left.
    rng: Rng,
    shove_left: u32,
    /// The walking speed the imitating walker is asked to change to (m/s).
    pace: f64,
}

impl Env {
    /// Control period: one action is held for this long.
    pub const DT: f64 = 1.0 / 60.0;
    /// Episodes are cut off after this many steps (10 s).
    pub const MAX_STEPS: u32 = 600;

    pub fn new(task: Task) -> Env {
        let mut env = Env {
            robot: Robot::new(Vec::new()),
            task,
            steps: 0,
            rng: Rng::new(0),
            shove_left: 0,
            pace: 0.0,
        };
        env.reset(0);
        env
    }

    /// Joints that have a motor, in order. The action has one entry for each.
    fn motors(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.robot.links.len()).filter(|&i| self.robot.links[i].gear > 0.0)
    }

    pub fn action_size(&self) -> usize {
        match self.task {
            Task::Imitate => IMITATE_ACTIONS,
            _ => self.motors().count(),
        }
    }

    /// Starts a new episode from a slightly random state fixed by `seed`. Returns the first observation.
    pub fn reset(&mut self, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut r = match self.task {
            Task::Balance => robot::cart_pole(),
            Task::Hop => robot::hopper(),
            Task::Reach => robot::arm(),
            Task::Imitate => robot::walker(),
        };
        // The agent drives the motors, not the built-in controller; except when imitating,
        // where it corrects the controller.
        r.controller = None;
        match self.task {
            Task::Balance => r.q[1] = 0.0, // upright
            Task::Hop => r.q[1] = 1.0,     // standing on the ground
            Task::Imitate => {
                // Start anywhere in the stride, as DeepMimic does, so every part gets practised.
                r.controller = Some(robot::imitate);
                r.adjust = vec![0.0; IMITATE_ACTIONS];
                // At any walking pace up to a brisk 2 m/s, or a quarter of the time from
                // standing still, to learn to step off.
                if rng.uniform() < -0.5 {
                    r.command = 0.0;
                    robot::start_in_stride(&mut r, 0.0);
                    r.command = WALKING_PACE.0
                        + (WALKING_PACE.1 - WALKING_PACE.0) * (0.5 + 0.5 * rng.uniform());
                } else {
                    r.command = WALKING_PACE.0
                        + (WALKING_PACE.1 - WALKING_PACE.0) * (0.5 + 0.5 * rng.uniform());
                    robot::start_in_stride(&mut r, 0.5 + 0.5 * rng.uniform());
                }
                self.pace = r.command;
            }
            Task::Reach => {
                let target = Vec3::new(
                    0.7 + 0.2 * rng.uniform(),
                    0.8 + 0.3 * rng.uniform(),
                    0.4 * rng.uniform(),
                );
                r.target = Some(target);
            }
        }
        for i in 0..r.q.len() {
            r.q[i] += 0.02 * rng.uniform();
            r.qd[i] += 0.02 * rng.uniform();
        }
        self.robot = r;
        self.steps = 0;
        (self.rng, self.shove_left) = (rng, 0);
        self.observe()
    }

    pub fn observe(&self) -> Vec<f64> {
        let r = &self.robot;
        match self.task {
            Task::Balance => {
                let angle = (r.q[1] + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                vec![r.q[0], r.qd[0], angle, r.qd[1]]
            }
            // The hopper's x position is left out, so a policy cannot depend on where it is.
            Task::Hop => r.q[1..].iter().chain(&r.qd).copied().collect(),
            Task::Imitate => robot::imitation_obs(r),
            Task::Reach => {
                let offset = r.target.unwrap() - self.tip();
                r.q.iter()
                    .chain(&r.qd)
                    .copied()
                    .chain(offset.to_array())
                    .collect()
            }
        }
    }

    fn tip(&self) -> Vec3 {
        let (o, rot) = *self.robot.frames(&self.robot.q).last().unwrap();
        o + rot * ARM_TIP
    }

    /// Applies `action` for one control period.
    pub fn step(&mut self, action: &[f64]) -> Step {
        assert_eq!(
            action.len(),
            self.action_size(),
            "action needs one entry per motor"
        );
        // NaN would poison the whole state; treat it as no command.
        let clean = |a: f64| if a.is_nan() { 0.0 } else { a.clamp(-1.0, 1.0) };
        if self.task == Task::Imitate {
            self.robot.adjust = action.iter().map(|&a| IMITATE_REACH * clean(a)).collect();
            // About every three seconds, asked for a new pace, which the request ramps to at
            // 0.8 m/s², about as quickly as people change pace: so it learns to speed up and
            // slow down. (Stopping is a separate standing controller, `robot::mimic_walk`.)
            if self.rng.uniform() > 1.0 - 1.0 / 90.0 {
                let u = 0.5 + 0.5 * self.rng.uniform();
                self.pace = WALKING_PACE.0 + (WALKING_PACE.1 - WALKING_PACE.0) * u;
            }
            let change = self.pace - self.robot.command;
            self.robot.command += change.clamp(-0.8 * Self::DT, 0.8 * Self::DT);
            // About every two seconds, a shove of up to 300 N either way on the chest for a
            // fifth of a second, so the policy learns to recover as well as to walk.
            if self.shove_left == 0 && self.rng.uniform() > 1.0 - 1.0 / 60.0 {
                let torso = self.robot.frames(&self.robot.q)[robot::WALKER_TORSO];
                let force = Vec3::X * 300.0 * self.rng.uniform();
                self.robot.external = vec![robot::Push {
                    link: robot::WALKER_TORSO,
                    at: torso.0 + torso.1 * Vec3::Y * 0.3,
                    force,
                }];
                self.shove_left = 12;
            }
            if self.shove_left > 0 {
                self.shove_left -= 1;
                if self.shove_left == 0 {
                    self.robot.external.clear();
                }
            }
        } else {
            let motors: Vec<usize> = self.motors().collect();
            for (&joint, &a) in motors.iter().zip(action) {
                self.robot.ctrl[joint] = clean(a);
            }
        }
        self.robot
            .step(Self::DT, if self.robot.ground { 16 } else { 8 });
        self.steps += 1;
        let r = &self.robot;
        let obs = self.observe();
        let (reward, terminated) = match self.task {
            Task::Balance => (1.0, obs[2].abs() > 0.4 || r.q[0].abs() > 2.4),
            Task::Hop => {
                let effort: f64 = action.iter().map(|a| a * a).sum();
                (
                    r.qd[0] + 1.0 - 0.001 * effort,
                    r.q[1] < 0.7 || r.q[2].abs() > 0.2,
                )
            }
            Task::Reach => (-(r.target.unwrap() - self.tip()).length(), false),
            Task::Imitate => {
                // DeepMimic's reward: near 1 when the pose matches, falling off smoothly.
                let pose = (-10.0 * robot::imitation_error(r)).exp();
                let speed = (-4.0 * (r.qd[0] - r.command).powi(2)).exp();
                (
                    0.7 * pose + 0.3 * speed,
                    r.q[1] < 0.7 || r.q[robot::WALKER_TORSO].abs() > 0.6,
                )
            }
        };
        Step {
            obs,
            reward,
            terminated,
            truncated: !terminated && self.steps >= Self::MAX_STEPS,
        }
    }
}

/// Learns a linear policy (`action = weights · observation`) by hill climbing:
/// nudge the best weights at random, run an episode, keep the nudge if it did
/// at least as well. Enough for cart-pole, and small enough to watch live.
pub struct Learner {
    pub env: Env,
    best: Vec<f64>,
    /// Return (total reward) of the best episode so far.
    pub best_return: f64,
    trial: Vec<f64>,
    trial_return: f64,
    /// Episodes finished.
    pub trials: u32,
    noise: f64,
    obs: Vec<f64>,
    rng: Rng,
}

impl Learner {
    pub fn new(task: Task, seed: u64) -> Learner {
        let mut env = Env::new(task);
        let obs = env.reset(seed);
        let mut rng = Rng::new(seed);
        let trial: Vec<f64> = (0..obs.len() * env.action_size())
            .map(|_| rng.uniform())
            .collect();
        Learner {
            env,
            best: trial.clone(),
            best_return: f64::NEG_INFINITY,
            trial,
            trial_return: 0.0,
            trials: 0,
            noise: 1.0,
            obs,
            rng,
        }
    }

    /// Advances the current episode by one step; starts the next one when it ends.
    pub fn tick(&mut self) {
        let n = self.obs.len();
        let action: Vec<f64> = self
            .trial
            .chunks(n)
            .map(|w| w.iter().zip(&self.obs).map(|(w, o)| w * o).sum())
            .collect();
        let step = self.env.step(&action);
        self.trial_return += step.reward;
        self.obs = step.obs;
        if step.terminated || step.truncated {
            // Search wider after a failure, narrower after a success.
            if self.trial_return >= self.best_return {
                (self.best, self.best_return) = (self.trial.clone(), self.trial_return);
                self.noise = (self.noise / 2.0).max(0.01);
            } else {
                self.noise = (self.noise * 2.0).min(2.0);
            }
            self.trials += 1;
            self.trial = self
                .best
                .iter()
                .map(|w| w + self.noise * self.rng.uniform())
                .collect();
            self.trial_return = 0.0;
            self.obs = self.env.reset(self.trials as u64);
        }
    }
}

/// The hopper policy trained by `python/train_hopper.py`: a neural network with two
/// hidden layers of 64 tanh units, from observation to motor commands.
pub fn hop_policy(obs: &[f64]) -> Vec<f64> {
    network(include_bytes!("hop_policy.bin"), obs, 2)
}

/// The walking policy trained by `python/train_hopper.py imitate`: corrections to the
/// imitation controller's targets (`Task::Imitate`), same network shape as the hopper's.
pub fn walk_policy(obs: &[f64]) -> Vec<f64> {
    network(include_bytes!("walk_policy.bin"), obs, IMITATE_ACTIONS)
}

/// Runs a network stored as raw little-endian f32: for each layer its weights (one row per
/// output), then its biases. Two hidden layers of 64 tanh units, then `outputs` linear ones.
fn network(stored: &[u8], obs: &[f64], outputs: usize) -> Vec<f64> {
    let mut weights = stored
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64);
    let mut x = obs.to_vec();
    for (size, squash) in [(64, true), (64, true), (outputs, false)] {
        let rows: Vec<f64> = weights.by_ref().take(size * x.len()).collect();
        assert_eq!(rows.len(), size * x.len(), "policy file is too short");
        x = rows
            .chunks(x.len())
            .zip(weights.by_ref())
            .map(|(row, bias)| row.iter().zip(&x).map(|(w, x)| w * x).sum::<f64>() + bias)
            .map(|sum| if squash { sum.tanh() } else { sum })
            .collect();
    }
    assert!(weights.next().is_none(), "policy file is too long");
    x
}

/// Runs the trained hopper policy, one episode after another, for watching and testing.
pub struct Player {
    pub env: Env,
    obs: Vec<f64>,
    /// Episodes started so far; also the seed of the current one.
    pub episode: u64,
    /// Time into the current episode, in seconds.
    pub seconds: f64,
}

impl Player {
    pub fn new(seed: u64) -> Player {
        let mut env = Env::new(Task::Hop);
        let obs = env.reset(seed);
        Player {
            env,
            obs,
            episode: seed,
            seconds: 0.0,
        }
    }

    /// Advances one control period; starts the next episode when this one ends.
    pub fn tick(&mut self) {
        let step = self.env.step(&hop_policy(&self.obs));
        self.seconds += Env::DT;
        self.obs = step.obs;
        if step.terminated || step.truncated {
            *self = Player::new(self.episode + 1);
        }
    }
}
