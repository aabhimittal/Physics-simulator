"""Run with:  python3 -m unittest discover python"""

import unittest

import physics_simulator as ps


class EnvTest(unittest.TestCase):
    def test_sizes(self):
        for task, obs, actions in [("balance", 4, 1), ("hop", 9, 2), ("reach", 11, 4)]:
            env = ps.Env(task)
            self.assertEqual((env.obs_size, env.action_size), (obs, actions))
            self.assertEqual(len(env.reset(1)), obs)

    def test_same_seed_same_episode(self):
        def run(seed):
            env = ps.Env("balance")
            env.reset(seed)
            return [env.step([0.3])[0] for _ in range(50)][-1]

        self.assertEqual(run(7), run(7))
        self.assertNotEqual(run(7), run(8))

    def test_doing_nothing_drops_the_pole(self):
        env = ps.Env("balance")
        env.reset(3)
        for _ in range(600):
            _, reward, terminated, truncated = env.step([0.0])
            self.assertEqual(reward, 1.0)
            self.assertFalse(truncated)
            if terminated:
                return
        self.fail("the pole never fell")

    def test_reach_is_cut_off_at_the_time_limit(self):
        env = ps.Env("reach")
        env.reset(1)
        flags = [env.step([0.0] * 4)[2:] for _ in range(600)]
        self.assertEqual(flags[-1], (False, True))
        self.assertTrue(all(f == (False, False) for f in flags[:-1]))

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            ps.Env("fly")
        with self.assertRaises(ValueError):
            ps.Env("balance").step([0.0, 0.0])

    def test_random_search_learns_to_balance(self):
        # The same trial-and-error idea as the Rust Learner, written against the Python API.
        import random

        rng = random.Random(1)
        env = ps.Env("balance")

        def episode(weights, seed):
            obs, steps = env.reset(seed), 0
            while True:
                obs, _, terminated, truncated = env.step([sum(w * o for w, o in zip(weights, obs))])
                steps += 1
                if terminated or truncated:
                    return steps

        best, best_steps, noise = [0.0] * 4, 0, 1.0
        for trial in range(300):
            candidate = [w + noise * rng.uniform(-1, 1) for w in best]
            steps = episode(candidate, trial)
            if steps >= best_steps:
                best, best_steps, noise = candidate, steps, max(noise / 2, 0.01)
            else:
                noise = min(noise * 2, 2.0)
            if best_steps == 600:
                return
        self.fail(f"best after 300 episodes was {best_steps} steps")


@unittest.skipUnless(hasattr(ps, "GymEnv"), "gymnasium and numpy are not installed")
class GymTest(unittest.TestCase):
    def test_passes_gymnasium_env_checker(self):
        from gymnasium.utils.env_checker import check_env

        for task in ps.TASKS:
            check_env(ps.GymEnv(task), skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
