from argparse import ArgumentParser
from typing import Dict

import gymnasium as gym

import env
from agents import *
from evaluation import ConvolutionEvaluation, create_check_final_filter


def run(env: gym.Env, agent: Agent, seed: int) -> Dict:
    prev_state, _ = env.reset(seed=seed)
    terminal = False
    info = None

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
    args.add_argument('--render', action='store_true', default=True)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    #agent = MCTSAgent(samples_limit=1_000, board_size=args.board_size, rollout_bound=6, confidence=0., silent=False)
    agent = AlphaBetaAgent(depth=1, evaluator=ConvolutionEvaluation(*create_check_final_filter()))
    opponent = RandomAgent(321)

    env = gym.make('Gomoku-v1', opponent=opponent.opponent_policy, board_size=args.board_size, render=args.render)

    result = run(env, agent, args.seed)
    print(f'{"Agent" if result["winner"] == 1 else "Opponent"} won in {len(result["moves"])} moves')
