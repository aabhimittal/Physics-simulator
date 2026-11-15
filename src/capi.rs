//! C interface to the reinforcement-learning environments, for bindings in other
//! languages. `python/physics_simulator.py` loads it with ctypes.
//!
//! An environment is an opaque pointer from `ps_env_new`, valid until `ps_env_free`.
//! Buffers are caller-allocated: `ps_env_obs_size` doubles for observations,
//! `ps_env_action_size` doubles for actions. Functions do nothing on a null pointer.

use crate::env::{Env, Task};

/// Creates an environment. `task`: 0 balance, 1 hop, 2 reach. Returns null for any other value.
#[unsafe(no_mangle)]
pub extern "C" fn ps_env_new(task: u32) -> *mut Env {
    let task = match task {
        0 => Task::Balance,
        1 => Task::Hop,
        2 => Task::Reach,
        _ => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(Env::new(task)))
}

/// # Safety
/// `env` must be null or a pointer from `ps_env_new` that has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ps_env_free(env: *mut Env) {
    if !env.is_null() {
        drop(unsafe { Box::from_raw(env) });
    }
}

/// # Safety
/// `env` must be null or a live pointer from `ps_env_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ps_env_obs_size(env: *const Env) -> u32 {
    unsafe { env.as_ref() }.map_or(0, |e| e.observe().len() as u32)
}

/// # Safety
/// `env` must be null or a live pointer from `ps_env_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ps_env_action_size(env: *const Env) -> u32 {
    unsafe { env.as_ref() }.map_or(0, |e| e.action_size() as u32)
}

/// Starts a new episode and writes the first observation to `obs`.
///
/// # Safety
/// `env` must be null or a live pointer from `ps_env_new`; `obs` must have room for `ps_env_obs_size` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ps_env_reset(env: *mut Env, seed: u64, obs: *mut f64) {
    if let Some(e) = unsafe { env.as_mut() } {
        let first = e.reset(seed);
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr(), obs, first.len()) };
    }
}

/// Applies `action` for one control period. Writes the observation and reward, and
/// returns flags: bit 0 set if the task failed (terminated), bit 1 if the time limit was reached (truncated).
///
/// # Safety
/// `env` must be null or a live pointer from `ps_env_new`; `action` must hold `ps_env_action_size`
/// doubles, `obs` must have room for `ps_env_obs_size` doubles, and `reward` for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ps_env_step(
    env: *mut Env,
    action: *const f64,
    obs: *mut f64,
    reward: *mut f64,
) -> u32 {
    let Some(e) = (unsafe { env.as_mut() }) else {
        return 0;
    };
    let action = unsafe { std::slice::from_raw_parts(action, e.action_size()) };
    let step = e.step(action);
    unsafe {
        std::ptr::copy_nonoverlapping(step.obs.as_ptr(), obs, step.obs.len());
        *reward = step.reward;
    }
    step.terminated as u32 | (step.truncated as u32) << 1
}
