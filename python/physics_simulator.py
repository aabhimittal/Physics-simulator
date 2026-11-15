"""Python bindings for the Physics-simulator reinforcement-learning environments.

Build the library first:  cargo build --release

    from physics_simulator import Env
    env = Env("balance")                  # or "hop", "reach"
    obs = env.reset(seed=1)
    obs, reward, terminated, truncated = env.step([0.0])

With gymnasium and numpy installed, `GymEnv("hop")` is a standard gymnasium.Env.
"""

import ctypes
import os
import sys
from pathlib import Path

TASKS = {"balance": 0, "hop": 1, "reach": 2}


def _load():
    name = {"darwin": "libphysics_simulator.dylib", "win32": "physics_simulator.dll"}.get(
        sys.platform, "libphysics_simulator.so"
    )
    default = Path(__file__).resolve().parent.parent / "target" / "release" / name
    path = os.environ.get("PHYSICS_SIMULATOR_LIB", default)
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as err:
        raise ImportError(
            f"Could not load {path}. Run `cargo build --release` in the repository, "
            "or set PHYSICS_SIMULATOR_LIB to the built library."
        ) from err
    doubles = ctypes.POINTER(ctypes.c_double)
    lib.ps_env_new.restype = ctypes.c_void_p
    lib.ps_env_new.argtypes = [ctypes.c_uint32]
    lib.ps_env_free.restype = None
    lib.ps_env_free.argtypes = [ctypes.c_void_p]
    for size in (lib.ps_env_obs_size, lib.ps_env_action_size):
        size.restype = ctypes.c_uint32
        size.argtypes = [ctypes.c_void_p]
    lib.ps_env_reset.restype = None
    lib.ps_env_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64, doubles]
    lib.ps_env_step.restype = ctypes.c_uint32
    lib.ps_env_step.argtypes = [ctypes.c_void_p, doubles, doubles, doubles]
    return lib


_lib = _load()


class Env:
    """One environment. Observations and actions are lists of floats; actions are in [-1, 1]."""

    def __init__(self, task):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose one of {sorted(TASKS)}")
        self._ptr = _lib.ps_env_new(TASKS[task])
        self.task = task
        self.obs_size = _lib.ps_env_obs_size(self._ptr)
        self.action_size = _lib.ps_env_action_size(self._ptr)
        self._obs = (ctypes.c_double * self.obs_size)()

    def __del__(self):
        if getattr(self, "_ptr", None):
            _lib.ps_env_free(self._ptr)
            self._ptr = None

    def reset(self, seed=0):
        """Starts a new episode. The same seed always gives the same start."""
        _lib.ps_env_reset(self._ptr, seed, self._obs)
        return list(self._obs)

    def step(self, action):
        """Returns (observation, reward, terminated, truncated)."""
        if len(action) != self.action_size:
            raise ValueError(f"{self.task} needs {self.action_size} action values, got {len(action)}")
        reward = ctypes.c_double()
        flags = _lib.ps_env_step(
            self._ptr, (ctypes.c_double * self.action_size)(*action), self._obs, ctypes.byref(reward)
        )
        return list(self._obs), reward.value, bool(flags & 1), bool(flags & 2)


try:
    import gymnasium
    import numpy as np
except ImportError:  # the plain Env above works without them
    gymnasium = None

if gymnasium is not None:

    class GymEnv(gymnasium.Env):
        """The same environments behind the standard Gymnasium API."""

        def __init__(self, task="balance"):
            self._env = Env(task)
            self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (self._env.obs_size,), np.float64)
            self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (self._env.action_size,), np.float64)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            # Each episode gets its own start, drawn from Gymnasium's seeded generator.
            episode_seed = int(self.np_random.integers(0, 2**63))
            return np.array(self._env.reset(episode_seed)), {}

        def step(self, action):
            obs, reward, terminated, truncated = self._env.step([float(a) for a in action])
            return np.array(obs), reward, terminated, truncated, {}
