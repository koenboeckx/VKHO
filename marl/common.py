import numpy as np
import torch

def preprocess(states):
    """
    Process the 'state' such that it can serve
    as input to the NN model.
    
    :param states:  list of states
    :return: tensor 
    """
    size = int(np.sqrt(len(states[0].board)))
    tensor = torch.zeros((len(states), 1, size, size))
    for idx, state in enumerate(states):
        board = state.board
        for i in range(size):
            for j in range(size):
                if board[size*i + j] != -1:
                    tensor[idx, 0, i, j] = int(board[size*i + j][-1]) + 1
    return tensor