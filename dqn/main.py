import gym
import torch.optim as optim
import os
import time
import argparse
import json

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learning
from utils.gym import get_env, get_wrapper_by_name
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

def main(env, num_timesteps, output_dir, 
         replay_buffer_size, batch_size, gamma, learning_starts, 
         learning_freq, frame_history_len, target_update_freq, learning_rate, alpha, eps):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learning(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        learning_starts=learning_starts,
        learning_freq=learning_freq,
        frame_history_len=frame_history_len,
        target_update_freq=target_update_freq,
        output_dir=output_dir,
        init_output_dir=True
    )

def get_args():
    parser = argparse.ArgumentParser(description='DQN params')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--replay_buffer_size', type=int, default=REPLAY_BUFFER_SIZE)
    parser.add_argument('--learning_starts', type=int, default=LEARNING_STARTS)
    parser.add_argument('--learning_freq', type=int, default=LEARNING_FREQ)
    parser.add_argument('--frame_history_len', type=int, default=FRAME_HISTORY_LEN)
    parser.add_argument('--target_update_freq', type=int, default=TARGER_UPDATE_FREQ)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--alpha', type=float, default=ALPHA)
    parser.add_argument('--eps', type=float, default=EPS)
    parser.add_argument('--task', type=int, default=PONG_TASK)
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')
    seed = 0 

    # Run training
    args = get_args()
    args_json = json.dumps(args)
    with open(os.path.join(output_dir, 'args.json')) as f:
        f.write(args_json)

    task = benchmark.tasks[args['task']]
    del args['task']

    # output directory
    timestamp = str(time.time()).split('.')[0]
    output_dir = os.path.join('results', timestamp)
    os.system(f"mkdir -p {output_dir}")
    print(f"output directory: {output_dir}")

    env = get_env(task, seed, output_dir)
    main(env, task.max_timesteps, output_dir, **args)
