# nn-gomoku

## Installation

1. Clone this repository:

```bash
cd $PROJECT_DIR
git clone git@github.com:Thematiq/nn-gomoku.git
cd nn-gomoku
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training and evaluation

The basic script which launches the game between two agents is `play.py` in the `game` directory. The most important 
lines of code from this script are discussed below:

```python
agent = RandomAgent()
```

This line creates an instance an agent which we want to train or evaluate. 

```python
opponent = RandomAgent()
```

This line creates an instance of an opponent agent. **Attention!** The opponent agent must be already trained and ready 
to play. You can use the static `load` method to restore the agent's state from the file.

```python 
env = gym.make('Gomoku-v1', opponent=opponent.opponent_policy, board_size=args.board_size, render=args.render)
```

This code creates an instance of the Gym environment. The `opponent_policy` method is a method of the opponent agent
which returns the next move of the opponent. The `render` flag determines whether the game should be rendered or not.

The `run` method is used to play the game and returns game information that contains the moves of both players and the
result of the game.

```python
prev_state, _ = env.reset(seed=args.seed)
action = agent.act(prev_state)
```

The initial state of the environment can be obtained by calling the `reset` method. The `seed` parameter is used to set
the random seed. The `act` method is used to get the first action of the agent.

```python
while not terminal:
    state, reward, terminal, _, info = env.step(action)
    agent.update(prev_state, action, reward, state, terminal)
    action = agent.act(state)
    prev_state = state
```

The agent-environment interaction is performed in the `while` loop. The `env.step` method is used to perform the action 
in the environment. The `update` method is used to update the agent's state and the `act` method is used to get the next 
action. **Attention!** If you want to use `run` method to evaluate the agent, you must comment out or delete the line
in which the `update` method is called.

### Agent implementation

The agent implementations are located in the `agents` directory. The `Agent` class is an abstract class which defines 
the interface of the agent. Every agent must implement three methods: `__init__`, `update` and `act`. 

```python
def __init__(self, ..., seed):
    ...
```

The `__init__` method is used to initialize the agent's state. The `seed` parameter is used to set the random seed.

```python
def update(self, prev_state, action, reward, state, terminal):
    ...
```

The `update` method is used to update the agent's state after performing the action in the environment and receiving 
the reward (e.g., update the neural network weights). The state of the environment is represented by a two-dimensional
matrix of shape `(board_size, board_size)`. The `action` parameter is an integer in the range `[0, board_size ** 2)`. 
The `reward` parameter is a float and the `terminal` parameter is a boolean value.

```python
def act(self, state):
    ...
```

The `act` method is used to get the next action based on the current state of the environment. The `state` parameter is
a two-dimensional matrix of shape `(board_size, board_size)`. The method must return an integer in the range 
`[0, board_size ** 2)`.

**Attention!** All three methods must be implemented in the agent class, even if they are not used.

The `Agent` class also defines the `opponent_policy` method which is used to set the agent as an opponent in the Gym
environment. The `opponent_policy` method returns the next move of the opponent. The `load` and `save` methods are used
to load and save the agent's state. You have to implement these methods in your agent if the agent has some state which 
should be saved and restored (e.g., neural network weights).

## Notes

The agent plays as the black player ('X'). The opponent plays as the white player ('O'). The board has values {-1, 0, 1}
where -1 represents the black player ('X'), 0 represents an empty field and 1 represents the white player ('O').
