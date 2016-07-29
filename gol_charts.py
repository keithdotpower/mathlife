#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

"""
This script was used to generate the charts used in the article, so pretty messy
"""


def sigmoid(X):
  """Calculate sigmoid for vector of values"""
  return 1 / (1 + np.exp(-X))

def show_two_points_on_line(x1, x2, x1_label = None, x2_label = None):
    if x1_label is None:
        x1_label = str(x1)
    if x2_label is None:
        x2_label = str(x2)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xl = int(x1) - 5
    xr = int(x2) + 5
    xs = range(xl, xr + 1)
    ys = [0] * len(xs)

    plt.grid()
    plt.plot(xs, ys)
    plt.axis([xl, xr, -1, 1])

    plt.plot([x1], [0], 'ro')
    ax.annotate(x1_label, xy=(x1, .05), textcoords='data')
    plt.plot([x2], [0], 'ro')
    ax.annotate(x2_label, xy=(x2, .05), textcoords='data')

show_two_points_on_line(-6, 0)
show_two_points_on_line(0, 36)
show_two_points_on_line(-0.5, 35.5)
show_two_points_on_line(-10.5, 45.5, '-0.5 - T', '35.5 + T')

# show sigmoid curve with thresholds T
xs = range(-20, 20)
ys = [sigmoid(x) for x in xs]

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(xs, ys)
plt.axis([-20, 20, -1, 2])

# Just set T = 10 for now
T = 10

plt.plot([-T], [sigmoid(-T)], 'ro')
xy = (-T, sigmoid(-T) + 0.1)
ax.annotate('(-T, 0.0 + e)', xy=xy, textcoords='data')


plt.plot([T], [sigmoid(T)], 'ro')
xy = (T, sigmoid(T) + 0.1)
ax.annotate('(T, 1.0 - e)', xy=xy, textcoords='data')

plt.grid()
plt.show()

