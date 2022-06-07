from re import S
import sys
import gym
import numpy as np
from collections import defaultdict
import random
from schedule import LinearSchedule

env = gym.make('Blackjack-v0')

# this is all plotting stuff :/
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator

matplotlib.style.use('ggplot')

ACTIONS = { 'stand' : 0, 'hit' : 1}
def generate_episode(env):
    episode = []
    state = env.reset()
    while True:
        probs = [0.75, 0.25] if state[0] > 18 else [0.25, 0.75]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    states, actions, rewards = zip(*episode)
    print('states:  ',states)
    print('actions: ', actions)
    print('rewards: ', rewards)
    print()
    return episode


def get_epsilon(exploration, t):
    eps = exploration.value(t)
    return eps

def apply_greedy_eps_policy(Q, state, state_count, action_size, exploration, t):
    random_action = random.randint(0, action_size - 1)
    best_action = np.argmax(Q[state])
    epsilon = get_epsilon(exploration, t)
    action = np.random.choice([best_action, random_action], p=[1. - epsilon, epsilon])
    return action

def evaluate_policy(Q, episodes=10000):
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        
        done = False
        while not done:
            action = np.argmax(Q[state])
            
            state, reward, done, _ = env.step(action=action)
            
        if reward > 0:
            wins += 1
        
    return wins / episodes


def sarsa(episodes, alpha, gamma=1.):

    # this is our value function, we will use it to keep track of the "value" of being in a given state
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # to decide what action to take and calculate epsilon we need to keep track of how many times we've
    # been in a given state and how often we've taken a given action when in that state
    state_count = defaultdict(float)
    init_state_count = defaultdict(float)

    # for keeping track of our policy evaluations (we'll plot this later)
    exploration = LinearSchedule(episodes, 0.1)
    for t in range(episodes):
        new_state = env.reset()
        init_state_count[new_state] += 1
        new_action = apply_greedy_eps_policy(Q, new_state, state_count[new_state], env.action_space.n, exploration, t)

        done = False
        while not done:
            s_t = new_state
            state_count[s_t] += 1
            a_t = new_action

            new_state, r_t, done, _ = env.step(action=a_t)
            r_t = 1 if r_t > 0 else 0
            new_action = apply_greedy_eps_policy(Q, new_state, state_count[new_state], env.action_space.n, exploration, t)

            # update step
            Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r_t + gamma * Q[new_state][new_action] - Q[s_t][a_t])

            # enforce reward boundaries
            Q[s_t][a_t] = min(Q[s_t][a_t], 1)
            Q[s_t][a_t] = max(Q[s_t][a_t], 0)

    init_state_count = { s_t : c / episodes for s_t, c in init_state_count.items()}
    init_state_count = defaultdict(float, init_state_count)
    return Q, init_state_count

def apply_td0_policy(s_t):
    if s_t[0] >= 18: 
        return ACTIONS['stand']
    else:
        return ACTIONS['hit']

def td_0(episodes, alpha, gamma=1., win_probs_rewards=True):

    # this is our value function, we will use it to keep track of the "value" of being in a given state
    V = defaultdict(lambda: 0)

    # to decide what action to take and calculate epsilon we need to keep track of how many times we've
    # been in a given state and how often we've taken a given action when in that state
    init_state_count = defaultdict(float)

    for i in range(episodes):
        # evaluating a policy is slow going, so let's only do this every 1000 games
        s_t = env.reset()
        done = False
        while not done:
            init_state_count[s_t] += 1
            a_t = apply_td0_policy(s_t)
            new_state, r_t, done, _ = env.step(action=a_t)
            r_t = 1 if r_t > 0 else 0

            # update step
            V[s_t] = V[s_t] + alpha * (r_t + gamma * V[new_state] - V[s_t])

            # enforce reward boundaries
            V[s_t] = min(V[s_t],1) 
            V[s_t] = max(V[s_t],0) 
            s_t = new_state

    init_state_count = { s_t : count / episodes for s_t, count in init_state_count.items()}
    init_state_count = defaultdict(float, init_state_count)
    return V, init_state_count

def plot_value_function(V, vmin=-1.0, vmax=1.0, title='Value Function'):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    # Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Player sum')
        ax.set_ylabel('Dealer showing')
        ax.set_zlabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.view_init(ax.elev, 120)
        fig.colorbar(surf)
        plt.show()
    
    plot_surface(X, Y, Z_noace, title)
    #plot_surface(X, Y, Z_ace, "value function - usable ace")

def plot_Q_function(Q, title="Q Function (state-action function)"):
    V = defaultdict(float)
    for state, action_rewards in Q.items():
        r1, r2 = action_rewards
        action_value = np.max([r1, r2])
        V[state] = action_value
    plot_value_function(V)    

def calc_winning_prob(V, init_state_probs):
    winning_prob = 0
    for state in V:
        winning_prob += init_state_probs[state] * V[state]
    return winning_prob

def calc_greedy_V(Q):
    optimal_policy = defaultdict(float)
    V = defaultdict(float)
    for state, action_rewards in Q.items():
        r1, r2 = action_rewards
        V[state] = np.max([r1, r2])
        action = np.argmax([r1, r2])
        if action == 0:
            action = -1
        optimal_policy[state] = action
    return V, optimal_policy

def play_sarsa(episodes):
    episodes = int(episodes)
    for alpha in [1e-4]:
        label = f'alpha = {alpha}'
        print(f'alpha = {label}: Training...')
        Q_srs, init_state_probs = sarsa(episodes=episodes, alpha=alpha)
        # plt.plot([i * 1000 for i in range(len(evaluations))], evaluations, label=label)
        print(f'alpha = {label}: Finished Training')

        greedy_V, optimal_policy = calc_greedy_V(Q_srs)
        winning_prob = calc_winning_prob(greedy_V, init_state_probs)
        print(f'winning_prob = {winning_prob}')
        print()

        vmin=min(greedy_V.values())
        vmax=max(greedy_V.values())
        plot_value_function(greedy_V, vmin=vmin, vmax=vmax, title='V(s)')

        vmin=0
        vmax=max(init_state_probs.values())
        plot_value_function(init_state_probs, vmin=vmin, vmax=vmax, title='Init States prob')

        vmin=-1
        vmax=1
        plot_value_function(optimal_policy, vmin=vmin, vmax=vmax, title='action')

def play_td(episodes):
    episodes = int(episodes)
    for alpha in [1e-4]:
        label = f'alpha = {alpha}'
        print(f'alpha = {label}: Training...')
        V_td0, init_state_probs = td_0(episodes=episodes, alpha=alpha)
        print(f'alpha = {label}: Finished Training')

        winning_prob = calc_winning_prob(V_td0, init_state_probs)
        print(f'winning_prob = {winning_prob}')
        print()

        vmin=min(V_td0.values())
        vmax=max(V_td0.values())
        plot_value_function(V_td0, vmin=vmin, vmax=vmax, title='V(s)')

        vmin=0
        vmax=max(init_state_probs.values())
        plot_value_function(init_state_probs, vmin=vmin, vmax=vmax, title='Init States prob')
    return V_td0

episodes=1e6 # we need alot of episodes in order to sample our init-state dist
play_td(episodes)

episodes=1e6 # we need alot of episodes in order to sample our init-state dist
play_sarsa(episodes)
