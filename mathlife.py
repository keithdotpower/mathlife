#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

"""
This script implements Conway's Game of Life using maths.

There's an article that describes what this does. Link and PDF to follow.
"""


def make_neighbour_matrix(n):
    K = n * n

    # these are the values to add to a cell's row and col to get each neighbour (neater than using two loops)
    neighbour_offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    C = np.zeros(shape=(K, K))
    for r in range(n):
        for c in range(n):
            row_num_in_C = r * n + c

            # for each cell we want to get the indices for the 8 neighbours and set those columns in the current row to 1
            for offset in neighbour_offsets:
                nr = (r + offset[1]) % n
                nc = (c + offset[0]) % n
                n_index = nr * n + nc
                C[row_num_in_C][n_index] = 1
    return C


def update_cells(X, C):
    N = np.dot(C, X)
    X_prime = N * (N - 1) * (N - 4) * (N - 5) * (N - 6) * (N - 7) * (N - 8) * (N + X -2)
    return X_prime


def from_grid(G):
    len = G.shape[0] * G.shape[1]
    return G.reshape(len, 1)


def to_grid(V, r, c):
    return X.reshape(r, c)


# used to hold reference to open plot so can close it
fig = None

def handle_key_press(event):
    plt.close(fig)
    if event.key == 'escape':
        sys.exit(0)


def display(grid):
    # print(grid)
    global fig
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', handle_key_press)
    plt.imshow(grid, extent=(0, grid.shape[1], 0, grid.shape[0]),
               interpolation='nearest', cmap='Greys_r')
    plt.show()



def sigmoid(X):
  """Calculate sigmoid for vector of values"""
  return 1 / (1 + np.exp(-X))


def logit(x):
  """Inverse sigmoid for a single value x"""
  return math.log(x / (1 - x))


def restrict_ranges(X, epsilon, delta, T):
  """Given that X contains values near (Within distance epsilon) 0 and N between (2 and 720)
     Map them so that the ones near 0 go to [0, delta) and the ones not near 0 go to (1 - delta, 1]"""

  # first square so the large negative values becom positive, numbers around 0 will stay around 0
  X_prime = X ** 2

  # translate so numbers are near -.5 and > .5, then scale so numbers are > T from 0
  X_prime = (X_prime - .5) * 2 * T

  # take sigmoid so numbers will be back within acceptable range
  return sigmoid(X_prime)


if __name__ == '__main__':
    # define a Toad, Blinker and Glider structure (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    grid = np.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ])


    n = grid.shape[0] # take n from the grid we defined (assuming square)
    C = make_neighbour_matrix(n)

    epsilon = .1
    delta = epsilon / (8 ** 7)
    T = logit (1 - delta)

    print('Press ESC to quit, any other key to proceed to next step')

    # make state vector from grid
    X = from_grid(grid)

    # display state vector as grid and then loop doing updates and displaying them
    display(to_grid(X, n, n))

    while True:
        X = update_cells(X, C)
        X = restrict_ranges(X, epsilon, delta, T)
        display(to_grid(X, n, n))
