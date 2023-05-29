from argparse import ArgumentParser
from typing import Dict

import gymnasium as gym

import env
from agents import *


def run(env: gym.Env, agent: Agent, seed: int) -> Dict:
    prev_state, _ = env.reset(seed=seed)
    terminal = False

    while not terminal:
        action = agent.act(prev_state)
        state, reward, terminal, _, info = env.step(action)
        agent.update(prev_state, action, reward, state, terminal)
        prev_state = state

    return info


if __name__ == '__main__':
    gym.logger.set_level(40)

    args = ArgumentParser()
    args.add_argument('--board_size', type=int, default=9)
    args.add_argument('--render', action='store_true', default=False)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    agent = MCTSAgent(samples_limit=1000, board_size=9)
    opponent = RandomAgent(321)

    env = gym.make('Gomoku-v1', opponent=opponent.opponent_policy, board_size=args.board_size, render=args.render)

    result = run(env, agent, args.seed)
    print(f'{"Agent" if result["winner"] == 1 else "Opponent"} won in {len(result["moves"])} moves')
