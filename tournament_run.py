from game.tournament import Tournament
from agents import DQN, MCTSAgent, AlphaBetaAgent
from glob import glob
from evaluation import ConvolutionEvaluation, create_check_final_filter


tournament = Tournament(tournaments_num=20, players_from_population=2, board_size=9, c=1.41)

BOARD_SIZE = 9
RANDOM_STATE = 42

dqn_params = [
    # 4
    {
        'board_size': BOARD_SIZE,
        'seed': RANDOM_STATE,
        'sizes': (BOARD_SIZE ** 2, 22, 6, 22, BOARD_SIZE ** 2),
        'gamma': 0.5248139041127986,
        'learning_rate': 0.0006286808344418563,
        'epsilon': 0.9053617007397796,
        'epsilon_decay': 0.9981440158165802,
        'soft_update': 0.0021098721555291326,
        'capacity': 14623,
        'experience_replay_steps': 5
    },

    # 9
    {
        'board_size': BOARD_SIZE,
        'seed': RANDOM_STATE,
        'sizes': (BOARD_SIZE ** 2, 38, 10, 4, 10, 38, BOARD_SIZE ** 2),
        'gamma': 0.5073311667709228,
        'learning_rate': 0.005625500500086983,
        'epsilon': 0.9023464500248064,
        'epsilon_decay': 0.9764876491102636,
        'soft_update': 0.01377101929862648,
        'capacity': 10916,
        'experience_replay_steps': 9
    },

    # 5
    {
        'board_size': BOARD_SIZE,
        'seed': RANDOM_STATE,
        'sizes': (BOARD_SIZE ** 2, 24, 6, 6, 24, BOARD_SIZE ** 2),
        'gamma': 0.5038973168464231,
        'learning_rate': 0.009799988630708788,
        'epsilon': 0.9601050968090074,
        'epsilon_decay': 0.9636742555530011,
        'soft_update': 0.02630466420390503,
        'capacity': 30992,
        'experience_replay_steps': 2
    },

    # 15
    {
        'board_size': BOARD_SIZE,
        'seed': RANDOM_STATE,
        'sizes': (BOARD_SIZE ** 2, 21, 6, 21, BOARD_SIZE ** 2),
        'gamma': 0.6637276796030837,
        'learning_rate': 0.0029581177390474132,
        'epsilon': 0.91641476959713,
        'epsilon_decay': 0.9881843411050839,
        'soft_update': 0.018096742648194813,
        'capacity': 1402,
        'experience_replay_steps': 4
    }
]


tournament.new_population(
    agent_type=AlphaBetaAgent,
    params_list=[
        {'depth': i,
         'evaluator': ConvolutionEvaluation(*create_check_final_filter())}
        for i in range(1, 3)
    ]
)

tournament.new_population(
    agent_type=MCTSAgent,
    params_list=[
        {'samples_limit': limit,
         'rollout_bound': rollout,
         'board_size': 9}
        for limit in [2_000, 4_000, 8_000, 16_000]
        for rollout in [2, 4]
    ]
)

tournament.new_population(
    agent_type=DQN,
    params_list=dqn_params,
    load_path='data/best_zoo_2/*'
)

tournament.run()

print('Best players:')

for i, player in enumerate(tournament.best_players(10), start=1):
    print(f'{i}. [{player.num_wins / player.num_wins:.3f}] {player}')
