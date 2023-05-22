from glob import glob
from typing import Dict, List

import numpy as np
import gymnasium as gym
import gym_gomoku
import torch
from gym_gomoku.envs.util import GomokuUtil
from tqdm import trange

from agents import *

gym.logger.set_level(40)


class Player:
    """
    Class that represents a player in a tournament. It is used to keep track of the
    number of wins and games of each agent. It also calculates the value of the agent
    using the UCB algorithm.

    Parameters
    ----------
    agent : Agent
        The agent to be used.
    params : Dict
        The parameters of the agent.
    c : float
        The exploration parameter.
    """

    def __init__(self, agent: Agent, params: Dict, c: float) -> None:
        self.agent = agent
        self.params = params

        self.num_wins = 0
        self.num_games = 0
        self.num_tournaments = 0

        self.c = c

    def __call__(self, state: np.ndarray, *_) -> int:
        return self.agent.act(state, is_training=False)

    def __str__(self) -> str:
        return f'{type(self.agent).__name__}({self.params})'

    def update(self, wins: int, games: int) -> None:
        self.num_wins += wins
        self.num_games += games
        self.num_tournaments += 1

    def value(self, total_tournaments: int) -> float:
        if self.num_games == 0:
            return 1e9

        return self.num_wins / self.num_games + self.c * np.sqrt(np.log(total_tournaments) / self.num_tournaments)


class Population:
    """
    Class that represents a population of agents. It is used to draw agents for a single
    tournament. The agents are drawn with probability proportional to their value.

    Parameters
    ----------
    c : float
        The exploration parameter.
    """

    def __init__(self, agent_type: type, c: float) -> None:
        self.population = []
        self.agent_type = agent_type
        self.c = c

    def new_player(self, params: Dict, load_path: str = None) -> None:
        if load_path is not None:
            self.population.append(Player(self.agent_type.load(load_path), params, self.c))
        else:
            self.population.append(Player(self.agent_type(**params), params, self.c))

    def draw(self, n: int, total_tournaments: int) -> List[Player]:
        values = [player.value(total_tournaments) for player in self.population]
        probs = values / np.sum(values)
        return np.random.choice(self.population, n, p=probs, replace=False).tolist()


class Tournament:
    """
    Class that represents a tournament. It is used to run a tournament between agents
    of different types. The populations of agents are created for each type of agent by
    specifying the agent type and the parameters of the agents. For each tournament, the
    agents are drawn from the population with probability proportional to their value.
    Value of an agent is calculated using the UCB algorithm with the number of wins and
    games of the agent. In each tournament, the agents play against each other and the
    number of wins and games are updated. After the tournament, the best agents are chosen
    from all the populations.

    Parameters
    ----------
    tournaments_num : int
        The number of tournaments.
    players_from_population : int
        The number of agents drawn from each population.
    board_size : int
        The size of the board.
    c : float
        The exploration parameter.
    """

    def __init__(self, tournaments_num: int, players_from_population: int, board_size: int, c: float) -> None:
        self.tournaments_num = tournaments_num
        self.players_from_population = players_from_population
        self.board_size = board_size
        self.c = c

        self.populations = []

    def new_population(self, agent_type: type, params_list: List[Dict], load_path: str = None) -> None:
        if load_path is None:
            load_path = [None] * len(params_list)
        else:
            load_path = glob(load_path)

        population = Population(agent_type, self.c)

        for params, load_path in zip(params_list, load_path):
            population.new_player(params, load_path)

        self.populations.append(population)

    def _play_game(self, player1: Player, player2: Player) -> bool:
        env = gym.make(f'Gomoku{self.board_size}x{self.board_size}-v0', opponent=player2)
        env.reset()

        action = np.random.choice(env.action_space.n)
        terminal = False

        while not terminal:
            state, _, terminal, _ = env.step(action)
            action = player1(state)

        return GomokuUtil().check_five_in_row(state)[1] == 'black'

    def run(self) -> None:
        for tournaments in trange(self.tournaments_num):
            players = []

            for population in self.populations:
                players += population.draw(self.players_from_population, tournaments)

            wins = [0] * len(players)

            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    if self._play_game(players[i], players[j]):
                        wins[i] += 1
                    else:
                        wins[j] += 1

            for player, win in zip(players, wins):
                player.update(win, self.players_from_population - 1)

    def best_players(self, n: int) -> List[Player]:
        players = []

        for population in self.populations:
            players += population.population

        values = [player.num_wins / player.num_games for player in players]
        return [players[i] for i in np.argsort(values)[::-1][:n]]


if __name__ == '__main__':
    tournament = Tournament(tournaments_num=100, players_from_population=3, board_size=9, c=0.1)
    tournament.new_population(RandomAgent, [{'seed': i + 100} for i in range(10)])
    tournament.new_population(RandomAgent, [{'seed': i + 200} for i in range(10)])
    tournament.new_population(RandomAgent, [{'seed': i + 300} for i in range(10)])
    tournament.run()

    print('Best players:')

    for i, player in enumerate(tournament.best_players(10), start=1):
        print(f'{i}. [{player.num_wins / player.num_wins:.3f}] {player}')