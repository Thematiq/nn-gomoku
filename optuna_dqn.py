import optuna
import gymnasium as gym
from random import randint
from game.play import run

from agents import RandomAgent, AlphaBetaAgent, DQN
from evaluation import ConvolutionEvaluation, create_check_final_filter


BOARD_SIZE = 9
RANDOM_STATE = 42

EVAL_RANDOM = 10
EVAL_MINMAX = 10


def run_series(model, opponent_type, times):
    wins = 0

    for _ in range(times):
        if opponent_type == 'random':
            opponent = RandomAgent(randint(1, 2 ** 30))
        elif opponent_type == 'minmax':
            opponent = AlphaBetaAgent(depth=1, evaluator=ConvolutionEvaluation(*create_check_final_filter()))
        env = gym.make('Gomoku-v1', opponent=opponent.opponent_policy, board_size=BOARD_SIZE, render=False)
        res = run(env, model, RANDOM_STATE)
        if res['winner'] == 1:
            wins += 1
    return wins


def evaluate_dqn(model):
    wr1 = run_series(model, 'random', 10) / 10
    wr2 = run_series(model, 'minmax', 10) / 10
    return (wr1 + wr2) / 2


def sample_dqn_params(trial):
    return {
        'board_size': BOARD_SIZE,
        'seed': RANDOM_STATE,
        'gamma': trial.suggest_float(0, 1),
        'learning_rate': trial.suggest_float(1e-5, 1e-2),
        'epsilon': trial.suggest_float(0, 1),
        'epsilon_decay': trial.suggest_float(0, 1),
        'soft_update': trial.suggest_float(0, 1),
        'capacity': trial.suggest_int(1e2, 5e4),
        'experience_replay_steps': trial.suggest_int(1, 10)
    }


def objective(trial):
    params = sample_dqn_params(trial)
    random_rounds = trial.suggest_int(5, 100)
    minmax_rounds = trial.suggest_int(5, 50)

    agent = DQN(**params)

    run_series(agent, 'random', random_rounds)
    run_series(agent, 'minmax', minmax_rounds)

    agent.save(f'data/zoo/{trial.number}.pt')

    return evaluate_dqn(agent)


study = optuna.create_study(
    storage="sqlite:///data/dqn.sqlite3",
    study_name="dqn",
    load_if_exists=True
)

study.optimize(objective, n_trials=100, n_jobs=1)
