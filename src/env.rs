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
}

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
        };
        env.reset(0);
        env
    }

    /// Joints that have a motor, in order. The action has one entry for each.
    fn motors(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.robot.links.len()).filter(|&i| self.robot.links[i].gear > 0.0)
    }

    pub fn action_size(&self) -> usize {
        self.motors().count()
    }

    /// Starts a new episode from a slightly random state fixed by `seed`. Returns the first observation.
    pub fn reset(&mut self, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut r = match self.task {
            Task::Balance => robot::cart_pole(),
            Task::Hop => robot::hopper(),
            Task::Reach => robot::arm(),
        };
        r.controller = None; // the agent drives the motors, not the built-in controller
        match self.task {
            Task::Balance => r.q[1] = 0.0, // upright
            Task::Hop => r.q[1] = 1.0,     // standing on the ground
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
        let motors: Vec<usize> = self.motors().collect();
        for (&joint, &a) in motors.iter().zip(action) {
            // NaN would poison the whole state; treat it as no command.
            self.robot.ctrl[joint] = if a.is_nan() { 0.0 } else { a.clamp(-1.0, 1.0) };
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
                    r.q[1] < 0.7 || r.q[2].abs() > 1.0,
                )
            }
            Task::Reach => (-(r.target.unwrap() - self.tip()).length(), false),
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
