from gymnasium.envs.registration import register


register(
    id=f'Gomoku-v1',
    entry_point='env.gomoku:GomokuEnv'
)
