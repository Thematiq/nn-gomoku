import optuna
import json
import gymnasium as gym
import numpy as np

from random import randint
from game.play import run

from agents import RandomAgent, AlphaBetaAgent, DQN, MCTSAgent
from evaluation import ConvolutionEvaluation, create_check_final_filter


BOARD_SIZE = 9
RANDOM_STATE = 42

EVAL_RANDOM = 5
EVAL_MINMAX = 10
EVAL_MCTS = 15


def run_series(model, opponent_type, times, is_training, mcts_rounds=10_000):
    wins = 0

    for _ in range(times):
        if opponent_type == 'random':
            opponent = RandomAgent(randint(1, 2 ** 30))
        elif opponent_type == 'minmax':
            opponent = AlphaBetaAgent(depth=1, evaluator=ConvolutionEvaluation(*create_check_final_filter()))
        elif opponent_type == 'mcts':
            opponent = MCTSAgent(samples_limit=mcts_rounds, rollout_bound=2, board_size=BOARD_SIZE)
        env = gym.make('Gomoku-v1', opponent=opponent.opponent_policy, board_size=BOARD_SIZE, render=False)
        res = run(env, model, RANDOM_STATE, is_training=is_training)
        if res['winner'] == 1:
            wins += 1
    return wins


def evaluate_dqn(model):
    wr1 = run_series(model, 'random', EVAL_RANDOM, is_training=False) / EVAL_RANDOM
    wr2 = run_series(model, 'minmax', EVAL_MINMAX, is_training=False) / EVAL_MINMAX
    wr3 = run_series(model, 'mcts', EVAL_MCTS, is_training=False) / EVAL_MCTS
    wr4 = run_series(model, 'mcts', EVAL_MCTS, is_training=False, mcts_rounds=20_000) / EVAL_MCTS
    return np.mean([wr1, wr2, wr3, wr4])


def sample_q_architecture(trial):
    layers = trial.suggest_int("no_layers", 3, 5)

    last_size = BOARD_SIZE ** 2

    in_layers = []

    for x in range(layers // 2):
        in_layers.append(trial.suggest_int(f"layer_{x}_size", last_size // 4, last_size // 2))
        last_size = in_layers[-1]

    if layers % 2 == 1:
        middle = (trial.suggest_int(f"layer_{layers // 2 + 1}_size", 2, last_size // 2), )
    else:
        middle = ()

    in_layers = tuple(in_layers)

    layout = (BOARD_SIZE ** 2, ) + in_layers + middle +\
        in_layers[::-1] + (BOARD_SIZE ** 2, )

    print(f"Architecture layout: {layout}")

    return layout


def sample_dqn_params(trial):
    return {
        'board_size': BOARD_SIZE,
        'seed': RANDOM_STATE,
        'sizes': sample_q_architecture(trial),
        'gamma': trial.suggest_float("gamma", 0.5, 1),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2),
        'epsilon': trial.suggest_float("epsilon", 0.9, 1),
        'epsilon_decay': trial.suggest_float("epsilon_decay", 0.95, 0.9999),
        'soft_update': trial.suggest_float("soft_update", 0.0001, 0.05),
        'capacity': trial.suggest_int("capacity", 1e2, 5e4),
        'experience_replay_steps': trial.suggest_int("replay_steps", 1, 10)
    }


def objective(trial):
    params = sample_dqn_params(trial)

    with open(f'data/zoo_params/{trial.number}.json', 'w') as f:
        json.dump(params, f)

    random_rounds = trial.suggest_int("random_rounds", 100, 1000)
    mcts_dumb = trial.suggest_int("mcts_rounds_dumb", 100, 500)
    minmax_rounds = trial.suggest_int("minmax_rounds", 100, 1000)
    mcts_rounds = trial.suggest_int("mcts_rounds", 50, 250)

    agent = DQN(**params)

    run_series(agent, 'random', random_rounds, is_training=True)
    run_series(agent, 'mcts', mcts_dumb, mcts_rounds=2_000, is_training=True)
    run_series(agent, 'minmax', minmax_rounds, is_training=True)
    run_series(agent, 'mcts', mcts_rounds, is_training=True)

    agent.save(f'data/zoo/v3/{trial.number}.pt')

    return evaluate_dqn(agent)


study = optuna.create_study(
    storage="sqlite:///data/dqn.sqlite3",
    study_name="dqn_v3_mcts",
    load_if_exists=True,
    direction="maximize"
)

study.optimize(objective, n_trials=1_000, n_jobs=2)
