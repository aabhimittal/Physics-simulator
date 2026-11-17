//! Checks the reinforcement-learning interface.

use physics_simulator::env::{Env, Learner, Task, hop_policy};

#[test]
fn observation_and_action_sizes_match_the_task() {
    for (task, obs, actions) in [
        (Task::Balance, 4, 1),
        (Task::Hop, 9, 2),
        (Task::Reach, 11, 4),
    ] {
        let mut env = Env::new(task);
        assert_eq!(env.reset(1).len(), obs, "{task:?} observation");
        assert_eq!(env.action_size(), actions, "{task:?} actions");
        assert_eq!(env.step(&vec![0.0; actions]).obs.len(), obs);
    }
}

#[test]
fn same_seed_gives_the_same_episode_and_different_seeds_differ() {
    let run = |seed| {
        let mut env = Env::new(Task::Balance);
        env.reset(seed);
        (0..50).map(|_| env.step(&[0.3]).obs).last().unwrap()
    };
    assert_eq!(run(7), run(7));
    assert_ne!(run(7), run(8));
}

#[test]
fn doing_nothing_fails_the_balance_and_hop_tasks() {
    for (task, actions) in [(Task::Balance, 1), (Task::Hop, 2)] {
        let mut env = Env::new(task);
        env.reset(3);
        let fell = (0..Env::MAX_STEPS).find_map(|_| {
            let s = env.step(&vec![0.0; actions]);
            assert!(
                !s.truncated,
                "{task:?} survived the whole episode with no control"
            );
            s.terminated.then_some(s.reward)
        });
        assert!(fell.is_some(), "{task:?} never terminated");
    }
}

#[test]
fn episodes_are_cut_off_at_the_time_limit() {
    let mut env = Env::new(Task::Reach);
    env.reset(1);
    for step in 1..=Env::MAX_STEPS {
        let s = env.step(&[0.0; 4]);
        assert!(!s.terminated && s.reward < 0.0);
        assert_eq!(s.truncated, step == Env::MAX_STEPS);
    }
}

#[test]
fn bad_actions_are_clamped_not_propagated() {
    let mut env = Env::new(Task::Balance);
    env.reset(1);
    let s = env.step(&[f64::NAN]);
    assert!(s.obs.iter().all(|x| x.is_finite()));
    let s = env.step(&[1e9]);
    assert!(s.obs.iter().all(|x| x.is_finite()));
}

#[test]
fn hill_climbing_learns_to_balance_the_pole() {
    for seed in 1..=5 {
        let mut learner = Learner::new(Task::Balance, seed);
        while learner.best_return < Env::MAX_STEPS as f64 {
            assert!(
                learner.trials < 300,
                "seed {seed}: best is {} steps after 300 episodes",
                learner.best_return
            );
            learner.tick();
        }
        println!(
            "seed {seed}: balanced for the full 10 s after {} episodes",
            learner.trials
        );
    }
}

#[test]
fn trained_policy_hops_the_full_ten_seconds_from_a_hundred_starts() {
    let mut distances = Vec::new();
    for seed in 1..=100 {
        let mut env = Env::new(Task::Hop);
        let mut obs = env.reset(seed);
        loop {
            let s = env.step(&hop_policy(&obs));
            let x = env.robot.q[0];
            assert!(!s.terminated, "seed {seed}: fell after {x:.1} m");
            obs = s.obs;
            if s.truncated {
                assert!(x > 30.0, "seed {seed}: only travelled {x:.1} m");
                distances.push(x);
                break;
            }
        }
    }
    distances.sort_by(f64::total_cmp);
    println!(
        "10 s runs: {:.1} m shortest, {:.1} m median, {:.1} m longest",
        distances[0], distances[50], distances[99]
    );
}
