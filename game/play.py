from argparse import ArgumentParser

import numpy as np
import gymnasium as gym
import gym_gomoku
from gym_gomoku.envs.util import GomokuUtil

from agents import *


def run(env, agent):
    prev_state = np.zeros_like(env.observation_space.shape)
    action = env.action_space.sample()
    terminal = False

    while not terminal:
        state, reward, terminal, info = env.step(action)
        agent.update(prev_state, action, reward, state, terminal)
        action = agent.act(state)
        prev_state = state

    return GomokuUtil().check_five_in_row(state)[1] == 'black'


if __name__ == '__main__':
    gym.logger.set_level(40)

    args = ArgumentParser()
    args.add_argument('--no-render', action='store_true', default=False)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    agent = RandomAgent()
    opponent = RandomAgent()

    env = gym.make('Gomoku15x15-v0', opponent=opponent.opponent_policy, render=not args.no_render)
    env.reset(seed=args.seed)

    if run(env, agent):
        print('Agent wins!')
    else:
        print('Opponent wins!')
