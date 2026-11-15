"""Train PPO (Stable-Baselines3) on one of the environments and report how it does.

    pip install stable-baselines3          # pulls in torch, gymnasium, numpy
    cargo build --release
    python3 python/train_hopper.py [task] [total_steps]     # defaults: hop 1000000

Prints the average result of 10 evaluation episodes before training and after
every tenth of the run: return, episode length in seconds, and for the hopper
the distance travelled.
"""

import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from physics_simulator import GymEnv

DT = 1 / 60  # control period, Env::DT in src/env.rs
EPISODES = 10


def evaluate(model, task):
    """Average return, duration (s) and forward distance (m) over EPISODES fresh episodes."""
    env = GymEnv(task)
    totals = [0.0, 0.0, 0.0]
    for episode in range(EPISODES):
        obs, _ = env.reset(seed=10_000 + episode)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            totals[0] += reward
            totals[1] += DT
            if task == "hop":
                totals[2] += obs[4] * DT  # observation 4 is the forward speed
    return [t / EPISODES for t in totals]


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "hop"
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    envs = make_vec_env(lambda: GymEnv(task), n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", envs, seed=0, device="cpu")
    print(f"{'steps':>9} {'return':>8} {'seconds':>8} {'metres':>7}")
    for chunk in range(11):
        if chunk:
            model.learn(total_steps // 10, reset_num_timesteps=False)
        ret, seconds, metres = evaluate(model, task)
        print(f"{model.num_timesteps:>9} {ret:>8.1f} {seconds:>8.2f} {metres:>7.2f}", flush=True)
    model.save(f"ppo_{task}")


if __name__ == "__main__":
    main()
