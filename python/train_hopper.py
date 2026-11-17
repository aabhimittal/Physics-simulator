"""Train PPO (Stable-Baselines3) on one of the environments and report how it does.

    pip install stable-baselines3          # pulls in torch, gymnasium, numpy
    cargo build --release
    python3 python/train_hopper.py [task] [total_steps] [seed]     # defaults: hop 1000000 0

Prints the average result of 10 evaluation episodes before training and after
every tenth of the run: return, episode length in seconds, and for the hopper
the distance travelled.

The best checkpoint is saved as ppo_<task>.zip, and its network as ppo_<task>.bin.
To watch a new hopper policy in the browser, copy ppo_hop.bin over
src/hop_policy.bin and rebuild web/sim.wasm (see the README).
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


def export(model, path):
    """Writes the policy network as raw float32: for each layer its weights, then its biases.

    This is the format `hop_policy` in src/env.rs reads. It assumes the default
    MlpPolicy (two hidden layers of 64 tanh units, observations not normalised).
    """
    net = model.policy
    with open(path, "wb") as out:
        for layer in [*net.mlp_extractor.policy_net, net.action_net]:
            if hasattr(layer, "weight"):  # skips the tanh layers
                out.write(layer.weight.detach().numpy().astype("<f4").tobytes())
                out.write(layer.bias.detach().numpy().astype("<f4").tobytes())


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "hop"
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    envs = make_vec_env(lambda: GymEnv(task), n_envs=4, seed=seed, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", envs, seed=seed, device="cpu")
    best = float("-inf")
    print(f"{'steps':>9} {'return':>8} {'seconds':>8} {'metres':>7}")
    for chunk in range(11):
        if chunk:
            model.learn(total_steps // 10, reset_num_timesteps=False)
        ret, seconds, metres = evaluate(model, task)
        print(f"{model.num_timesteps:>9} {ret:>8.1f} {seconds:>8.2f} {metres:>7.2f}", flush=True)
        # PPO can collapse late in a run, so keep the best checkpoint, not the last.
        if ret > best:
            best = ret
            model.save(f"ppo_{task}")
            export(model, f"ppo_{task}.bin")


if __name__ == "__main__":
    main()
