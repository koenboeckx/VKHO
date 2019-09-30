"""
Initial attempt to implement EGT for symmetric games with 2 players.
Uses general replicator dynamics.
Visualition only enabled for 2-action games
"""

import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
ALPHA = 0.01

## Rock/Paper/Scissors

A = np.array([[ 0, -1,  1],     # actions = R | P | S
              [ 1,  0, -1],
              [-1,  1,  0]])

## Stag Hunt
A = np.array([[4, 1],
              [3, 3]])

## Prisoner's Dilemma
A = np.array([[3, 0],
              [5, 1]])

## Matching Pennies
""" A = np.array([[1, 0],
              [0, 1]]) """

def generate_trace(A, n_steps=1000):
    """
    A: payoff matrix for symmetric game
    n_steps: number of iteration steps 
    """
    m, n = A.shape
    B = A.T

    # generate initial point in unit simplex for both players
    x = np.random.dirichlet((1,)*m)
    y = np.random.dirichlet((1,)*m)

    trace_x, trace_y = [], []

    for i in range(n_steps):
        # replicator dynamics
        dx = x * (x.dot(A) - x.dot(A).dot(y))
        dy = y * (y.dot(B) - y.dot(B).dot(x))

        # Euler integration
        x += ALPHA * dx
        y += ALPHA * dy

        # Normalize (due to rounding errors)
        x /= sum(x)
        y /= sum(y)

        if DEBUG:
            print(i, ' x = ', x)
            print(i, ' y = ', y)

        # add to trace
        trace_x.append(x.copy())
        trace_y.append(y.copy())
    
    trace_x = np.array(trace_x)
    trace_y = np.array(trace_y)

    return trace_x, trace_y

for i in range(100):
    trace_x, trace_y = generate_trace(A)
    plt.plot(trace_x[:, 0], trace_y[:, 0])
plt.show()
