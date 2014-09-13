'''
    Author: Bryant Finney
    Date: 7 September 2014
'''

import operators as op
import numpy as np
import copy as cp
from matplotlib import pyplot as p

x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)
A = np.zeros((len(y), len(x)))

# create a vector field A = -sin(y) + j sin(x)
for row in A:
    row += np.cos(x * np.pi / np.max(np.abs(x)))
for col in A.T:
    col += np.cos(y * np.pi / np.max(np.abs(y)))

G = op.gradient(x, y, A)

ex = (np.min(y), np.max(y), np.min(x), np.max(x))
implot = p.imshow(A, origin="lower", extent=ex)
implot.set_cmap("hot")
p.colorbar()
p.xticks(x)
p.yticks(y)

# because of the numerical derivatives, points are shifted over by 1/2 dx
xd = x[:-1] + .5 * np.diff(x)
yd = y[:-1] + .5 * np.diff(y)

p.figure()
p.quiver(xd, yd, np.real(G), np.imag(G))
p.xticks(xd)
p.yticks(yd)
p.grid(True)

p.show()
