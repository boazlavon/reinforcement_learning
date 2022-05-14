import gym
import pickle
import torch.optim as optim
import os
import time
import argparse
import json
from itertools import count
import random

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_eval
from utils.gym import get_eval_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
import sys

PONG_TASK=3
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def main(env, num_timesteps, output_dir, frame_history_len, replay_buffer_size):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = LinearSchedule(1000000, 0.1)
    dqn_eval(
        env=env,
        q_func=DQN,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion, frame_history_len=frame_history_len,
        replay_buffer_size=replay_buffer_size,
        output_dir=output_dir)

def get_args():
    parser = argparse.ArgumentParser(description='DQN params')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frame_history_len', type=int, default=FRAME_HISTORY_LEN)
    parser.add_argument('--replay_buffer_size', type=int, default=REPLAY_BUFFER_SIZE)
    parser.add_argument('--task', type=int, default=PONG_TASK)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Run training
    args = get_args()
    print(f"output directory: {args['output_dir']}")
    print(f"seed: {args['seed']}")
    seed = args['seed'] 
    del args['seed']

    task = benchmark.tasks[args['task']]
    del args['task']

    env = get_eval_env(task, seed, args['output_dir'])
    main(env, task.max_timesteps, **args)
