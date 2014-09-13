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
A = np.zeros((len(y), len(x)), dtype=np.complex128)

r = np.sqrt(x ** 2 + y ** 2)

# create a vector field A = -sin(y) + j sin(x)
for i, row in enumerate(A):
    row += np.sin(x * np.pi / np.max(np.abs(x))) * 1j
    row += np.sign(y[i]) * (x ** 2) * 50
for i, col in enumerate(A.T):
    col -= np.sin(y * np.pi / np.max(np.abs(y)))
    col -= np.sign(x[i]) * (y ** 2) * 1j * 50
    
# create a vector field A = x + y
# for row in A:
#     row += x ** 2
# for col in A.T:
#     col += 1j * y ** 2
    
p.quiver(x, y, np.real(A), np.imag(A))
p.xticks(x)
p.yticks(y)
p.grid(True)

C = op.curl(x, y, A)

print np.max(C), np.min(C)

# because of the numerical derivatives, points are shifted over by 1/2 dx
xd = x[:-1] + .5 * np.diff(x)
yd = y[:-1] + .5 * np.diff(y)

ex = (np.min(yd), np.max(yd), np.min(xd), np.max(xd))
p.figure()
implot = p.imshow(C, origin="lower", extent=ex)
implot.set_cmap("hot")
p.colorbar()
p.xticks(xd)
p.yticks(yd)

p.show()
