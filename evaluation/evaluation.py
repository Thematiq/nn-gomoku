from abc import ABC, abstractmethod


class Evaluation(ABC):
    """
    Base class for evaluating a position.

    Contains a method evaluate used to evaluate a state based on what player is on move.
    """

    @abstractmethod
    def evaluate(self, state, player) -> float:
        """
        Method for evaluating a position, based on current player to move.
        :param state: current state - position
            State is a matrix with values:
                0 - empty field
                1 - field with white stone
                -1 - field with black stone
        :param player: current player on move
        :return: A number representing evaluation of the position.
        Value meaning:
            0.0 - state is equal
            positive number - shows white advantage, bigger number indicates bigger advantage for white
            negative number - shows black advantage, smaller number indicates bigger advantage for black
            +inf - white won the game
            -inf - black won the game
        """
        raise NotImplementedError
