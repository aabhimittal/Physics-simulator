"""Train PPO (Stable-Baselines3) on one of the environments and report how it does.

    pip install stable-baselines3          # pulls in torch, gymnasium, numpy
    cargo build --release
    python3 python/train_hopper.py [task] [total_steps] [seed] [resume.zip]   # defaults: hop 1000000 0

Prints the average result of 10 evaluation episodes before training and after
every tenth of the run: return, episode length in seconds, and for the hopper
the distance travelled (hop and imitate).

The best checkpoint is saved as ppo_<task>.zip, and its network as ppo_<task>.bin; the
latest as ppo_<task>_last.zip, which a later run can resume from (long runs, such as
walk3d's, go in pieces). imitate and walk3d start from the current planar walking policy
(see warm_start).
To watch a new hopper policy in the browser, copy ppo_hop.bin over
src/hop_policy.bin and rebuild web/sim.wasm (see the README).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from physics_simulator import GymEnv

DT = 1 / 60  # control period, Env::DT in src/env.rs
EPISODES = 10
SPEED_OBS = {"hop": 4, "imitate": 5, "walk3d": 5}  # which observation is the forward speed


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
            if task in SPEED_OBS:
                totals[2] += obs[SPEED_OBS[task]] * DT
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


def warm_start(model, task):
    """Starts a walking policy from the planar one in src/walk_policy.bin: its weights for the
    observations the two share (all of the old policy's for imitate, whose observation only
    grows; the first 21, `imitation_obs`, for walk3d) and its six corrections, zeros for the
    rest, so it begins doing what that policy did. Exploration noise starts about where the
    planar run's ended, not at PPO's default of 1, which would trip it."""
    raw = np.fromfile(Path(__file__).resolve().parent.parent / "src" / "walk_policy.bin", "<f4")
    old_inputs, rest = divmod(raw.size - (64 + 64 * 64 + 64 + 6 * 64 + 6), 64)
    assert rest == 0, "src/walk_policy.bin is not a walking policy"
    shared = 21 if task == "walk3d" else old_inputs
    shapes = [(64, old_inputs), (64,), (64, 64), (64,), (6, 64), (6,)]
    parts, at = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        parts.append(torch.tensor(raw[at : at + n].reshape(shape)))
        at += n
    net = model.policy
    first, second, out = net.mlp_extractor.policy_net[0], net.mlp_extractor.policy_net[2], net.action_net
    with torch.no_grad():
        first.weight.zero_()
        first.weight[:, :shared] = parts[0][:, :shared]
        first.bias.copy_(parts[1])
        second.weight.copy_(parts[2])
        second.bias.copy_(parts[3])
        out.weight.zero_()
        out.weight[:6] = parts[4]
        out.bias.zero_()
        out.bias[:6] = parts[5]
        net.log_std.fill_(-0.7)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "hop"
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    resume = sys.argv[4] if len(sys.argv) > 4 else None
    envs = make_vec_env(lambda: GymEnv(task), n_envs=4, seed=seed, vec_env_cls=SubprocVecEnv)
    if resume:
        model = PPO.load(resume, env=envs, device="cpu")
    else:
        model = PPO("MlpPolicy", envs, seed=seed, device="cpu")
        if task in ("imitate", "walk3d"):
            warm_start(model, task)
    record = Path(f"ppo_{task}.best")  # the best return so far, across resumed runs
    best = float(record.read_text()) if resume and record.exists() else float("-inf")
    print(f"{'steps':>9} {'return':>8} {'seconds':>8} {'metres':>7}")
    for chunk in range(11):
        if chunk:
            model.learn(total_steps // 10, reset_num_timesteps=False)
        ret, seconds, metres = evaluate(model, task)
        print(f"{model.num_timesteps:>9} {ret:>8.1f} {seconds:>8.2f} {metres:>7.2f}", flush=True)
        model.save(f"ppo_{task}_last")
        # PPO can collapse late in a run, so keep the best checkpoint, not the last.
        if ret > best:
            best = ret
            model.save(f"ppo_{task}")
            export(model, f"ppo_{task}.bin")
            record.write_text(str(best))


if __name__ == "__main__":
    main()
