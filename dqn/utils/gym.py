"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import gym
from gym import wrappers
from utils.seed import set_global_seeds
from utils.atari_wrapper import wrap_deepmind, wrap_deepmind_ram
import os
import time

def get_eval_env(task, seed, output_dir):
    env_id = task.env_id
    env = gym.make(env_id)
    set_global_seeds(seed)
    env.seed(seed)
    timestamp = str(time.time()).split('.')[0]
    gym_dir = os.path.join(output_dir, f'eval_gym_results_{seed}')
    print(f"eval_dir: {gym_dir}")
    env = wrappers.Monitor(env, gym_dir, force=False, mode='evaluation')
    env = wrap_deepmind(env)
    return env

def get_env(task, seed, output_dir, force=True):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)
    timestamp = str(time.time()).split('.')[0]
    expt_dir = f'{output_dir}/gym-results' 
    env = wrappers.Monitor(env, expt_dir, force=False, resume=True, mode='training')
    env = wrap_deepmind(env)

    return env

def get_ram_env(env, seed):
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind_ram(env)

    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
