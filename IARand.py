import numpy as np


class IARand:

    def __init__(self) -> None:
        pass
        
    def __repr__(self) -> str:
        return "I am an AI that choose a random move to play"

    # choose a random move in the moves available and returns it
    def choose_move(self, board):
        index_move = np.random.randint(0,board[-1]-1)
        return board[index_move]

    def test(self):
        pass