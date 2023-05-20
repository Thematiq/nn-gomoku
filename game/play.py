from argparse import ArgumentParser

import torch
import numpy as np
import gymnasium as gym
import gym_gomoku
from gym_gomoku.envs.util import GomokuUtil

from evaluation.convolution_evaluation import ConvolutionEvaluation, create_filter, Position

from agents import *


def run(env: gym.Env, agent: Agent) -> bool:
    prev_state = np.zeros(env.observation_space.shape)
    state = np.zeros(env.observation_space.shape)
    terminal = False

    while not terminal:
        action = agent.act(state)
        state, reward, terminal, info = env.step(action)
        agent.update(prev_state, action, reward, state, terminal)
        prev_state = state

    return GomokuUtil().check_five_in_row(state)[1] == 'black'


if __name__ == '__main__':
    gym.logger.set_level(40)

    args = ArgumentParser()
    args.add_argument('--no-render', action='store_true', default=False)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    filters = torch.concatenate([create_filter(5, 5, Position.VERTICAL),
                                 create_filter(5, 3, Position.VERTICAL)])
    mask = torch.tensor([[[5.]], [[3.]]])
    # evaluation = ConvolutionEvaluation(filters, mask)
#     evaluation = RandomEvaluation()

#     agent = RandomAgent(123)
#     agent = AlphaBetaAgent(depth=5, evaluator=evaluation)

    # agent = DQN(board_size=15, seed=args.seed)
    agent = MCTSAgent(samples_limit=100, board_size=9)
    # agent = AlphaBetaAgent(depth=2)
    opponent = MCTSAgent(samples_limit=250, board_size=9)
    # opponent = RandomAgent(123)

    env = gym.make('Gomoku9x9-v0', opponent=opponent.opponent_policy, render=not args.no_render)
    env.reset(seed=args.seed)
    np.random.seed(args.seed)

    if run(env, agent):
        print('Agent wins!')
    else:
        print('Opponent wins!')
