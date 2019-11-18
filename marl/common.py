import numpy as np
import torch

def preprocess(states):
    """
    Process the 'state' such that it can serve
    as input to the NN model.
    
    :param states:  list of states
    :return: tensor 
    """
    # TODO: improve state representation -> see preprocess_extended
    if not isinstance(states, list):
        raise TypeError('states should be list')
    size = int(np.sqrt(len(states[0].board)))
    tensor = torch.zeros((len(states), 1, size, size))
    for idx, state in enumerate(states):
        board = state.board
        for i in range(size):
            for j in range(size):
                if board[size*i + j] != -1:
                    tensor[idx, 0, i, j] = int(board[size*i + j][-1]) + 1
    return tensor

def preprocess_extended(states):
    """
    Process 'state' such that it can serve as input
    to the extended NN model.

    :param states: list of states
    :return: (tensor0, tensor1):
                tensor0: input to conv net
                tensor1: input to in_fc
    """
    if not isinstance(states, list) and not isinstance(states, tuple):
        raise TypeError('states should be list')
    size = int(np.sqrt(len(states[0].board)))
    tensor0 = torch.zeros((len(states), 1, size, size))
    tensor1 = torch.zeros((len(states), 8))
    for idx, state in enumerate(states):
        board = state.board
        for i in range(size):
            for j in range(size):
                if board[size*i + j] != -1:
                    tensor0[idx, 0, i, j] = int(board[size*i + j][-1]) + 1
        tensor1[idx, :] = torch.tensor(list(state.alive) + list(state.ammo))
    return (tensor0, tensor1)
    
def preprocess_gym(states):
    return torch.tensor(states).float(), torch.tensor(0) # dummy var for consistency
