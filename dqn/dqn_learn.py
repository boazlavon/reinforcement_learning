"""
This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
from re import U
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

from dqn_model import DQN, DQN_RAM

LAST_EPISODES_COUNT = 100
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

def dqn_learning(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    output_dir=None,
    init_output_dir=True
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, s_t, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            s_t = torch.from_numpy(s_t).type(dtype).unsqueeze(0) / 255.0
            # with torch.no_grad() variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                action = model(Variable(s_t))
                action = action.data.max(1)[1]
                return action
        else:
            # select random action
            return torch.IntTensor([[random.randrange(num_actions)]])

    def play_step(obs_t, t, target_net):
        obs_idx = replay_buffer.store_frame(obs_t) 
        s_t = replay_buffer.encode_recent_observation()
        a_t = select_epilson_greedy_action(target_net, s_t, t)
        next_obs, r_t, is_done, _ = env.step(a_t)
        replay_buffer.store_effect(obs_idx, a_t, r_t, is_done)
        if is_done:
            next_obs = env.reset()
        return next_obs
    
    def calc_loss(training_net, target_net, replay_buffer, batch_size):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
        # Convert numpy nd_array to torch variables for calculation
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

        if USE_CUDA:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        current_Q_values = training_net(obs_batch).gather(1, act_batch.unsqueeze(1))
        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = target_net(next_obs_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = rew_batch + (gamma * next_Q_values)
        target_Q_values = target_Q_values.unsqueeze(1)
        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        # Note: clipped_bellman_delta * -1 will be right gradient
        d_error = clipped_bellman_error * -1.0
        return current_Q_values, d_error

    # Initialize target q function and q function, i.e. build the model.
    ######
    statistics_output_path = os.path.join(output_dir, 'statistics.pkl')
    model_output_path = os.path.join(output_dir, 'model.pkl')
    if USE_CUDA:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    statistic = {
        "mean_episode_rewards": [],
        "best_mean_episode_rewards": [],
        "mean_loss": [],
        "exploration": [],
    }
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        training_net = DQN_RAM(in_features=env.observation_space.shape[0],
                num_actions=env.action_space.n).to(device)
        target_net = DQN_RAM(in_features=env.observation_space.shape[0],
                num_actions=env.action_space.n).to(device)
    else:
        training_net = q_func(in_channels=frame_history_len,
                num_actions=env.action_space.n).to(device)
        target_net = q_func(in_channels=frame_history_len,
                num_actions=env.action_space.n).to(device)

    try:
        with open(statistics_output_path, 'rb') as f:
            statistic = pickle.load(f)
        print(f"Load statistics from {statistics_output_path}")
    except:
        print(f"Load statistics from scratch")
    
    try:
        with open(os.path.join(output_dir, 'model.pkl'), 'rb') as f:
            training_net = pickle.load(f)
        print(f"Load training net from {model_output_path}")
    except:
        print(f"Load training net from scratch")
        target_net = q_func(in_channels=frame_history_len,
                num_actions=env.action_space.n).to(device)
    target_net.load_state_dict(training_net.state_dict())
    ######

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(training_net.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    mean_loss = float('inf')
    obs_t = env.reset()
    LOG_EVERY_N_STEPS = 10000
    d_error = None

    for t in count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### 2. Step the env and store the transition
        # At this point, "obs_t" contains the latest observation that was
        # recorded from the simulator. 
        # Here, your code needs to 
        # 1. store this observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.

        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!

        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.

        # Specifically, obs_t must point to the new latest observation.
        # Useful functions you'll need to call:

        # Note that you cannot use "obs_t" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. 
        # 
        # The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.

        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)
        #####
        next_obs = play_step(obs_t, t, target_net)
        obs_t = next_obs
        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and obs_t should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # Note: Move the variables to the GPU if avialable
            # 3.b: fill in your own code to compute the Bellman error. This requires
            # evaluating the current and next Q-values and constructing the corresponding error.
            # Note: don't forget to clip the error between [-1,1], multiply is by -1 (since pytorch minimizes) and
            #       maskout post terminal status Q-values (see ReplayBuffer code).
            # 3.c: train the model. To do this, use the bellman error you calculated perviously.
            # Pytorch will differentiate this error for you, to backward the error use the following API:
            #       current.backward(d_error.data.unsqueeze(1))
            # Where "current" is the variable holding current Q Values and d_error is the clipped bellman error.
            # Your code should produce one scalar-valued tensor.
            # Note: don't forget to call optimizer.zero_grad() before the backward call and
            #       optimizer.step() after the backward call.
            # 3.d: periodically update the target network by loading the current Q network weights into the
            #      target_Q network. see state_dict() and load_state_dict() methods.
            #      you should update every target_update_freq steps, and you may find the
            #      variable num_param_updates useful for this (it was initialized to 0)
            #####
            current_Q_values, d_error = calc_loss(training_net, target_net, replay_buffer, batch_size)

            # Clear previous gradients before backward pass
            # run backward pass
            # Perfom the update
            optimizer.zero_grad()
            current_Q_values.backward(d_error.data)
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network if num_param_updates % target_update_freq == 0: target_net.load_state_dict(training_net.state_dict())
            #####

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-1 * LAST_EPISODES_COUNT:])
            if (d_error is not None):
                mean_loss = d_error.squeeze().mean()
        if len(episode_rewards) > LAST_EPISODES_COUNT:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        statistic["mean_episode_rewards"].append(mean_episode_reward)
        statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
        statistic["mean_loss"].append(mean_loss)
        statistic["exploration"].append(exploration.value(t))

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print(f"timestamp: {t}")
            print(f"episodes {len(episode_rewards)}")
            print(f"mean reward ({LAST_EPISODES_COUNT} episodes) {mean_episode_reward:.3f}")
            print(f"best mean reward {best_mean_episode_reward:.3f}")
            print(f"mean loss {mean_loss:.3E}")
            print(f"exploration {exploration.value(t):.3E}")
            sys.stdout.flush()

            # Dump statistics to pickle
            with open(statistics_output_path, 'wb') as f:
                pickle.dump(statistic, f)
                print(f"Saved statistic to {statistics_output_path}")

            with open(model_output_path, 'wb') as f:
                pickle.dump(target_net, f)
                print(f"Saved model to {model_output_path}")
            print()
