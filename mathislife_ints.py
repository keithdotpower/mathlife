#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
This script implements Conway's Game of Life using maths.

This particular version cheats and uses non-arithmetic operations in restrict_ranges()

There's an article that describes what this does. Link and PDF to follow.
"""


def make_neighbour_matrix(n):
    K = n * n

    # these are the values to add to a cell's row and col to get each neighbour (neater than using two loops)
    neighbour_offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    # Take column vector of nums 1 - K-1 and turn into a matrix so we can read off the neighbours' indices
    index_mat = np.arange(K).reshape(n, n)

    C = np.zeros(shape=(K, K))
    for r in range(index_mat.shape[0]):
        for c in range(index_mat.shape[1]):
            row_num_in_C = r * index_mat.shape[0] + c

            # for each cell we want to get the indices for the 8 neighbours and set those columns in the current row to 1
            for offset in neighbour_offsets:
                nr = (r + offset[1]) % n
                nc = (c + offset[0]) % n
                n_index = index_mat[nr][nc]
                C[row_num_in_C][n_index] = 1

    return C


def update_cells(X, C):
    N = np.dot(C, X)
    X_prime = N * (N - 1) * (N - 4) * (N - 5) * (N - 6) * (N - 7) * (N - 8) * (N + X -2)
    return X_prime


def restrict_ranges(X):
    # CHEAT here and reset non-zero values to 1
    for x in range(X.shape[0]):
        if X[x] != 0:
            X[x] = 1
    return X


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

# make state vector from grid
X = from_grid(grid)

print('Press ESC to quit, any other key to proceed to next step')

# display state vector as grid and then loop doing updates and displaying them
display(to_grid(X, n, n))

while True:
    X = update_cells(X, C)
    X = restrict_ranges(X)
    display(to_grid(X, n, n))
