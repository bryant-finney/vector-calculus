'''
    Author: Bryant Finney
    Date: 8 September 2014
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
#     row += np.sin(x * np.pi / np.max(np.abs(x))) * 1j
    row += (x ** 2) * 50
for i, col in enumerate(A.T):
#     col -= np.sin(y * np.pi / np.max(np.abs(y)))
    col -= (y ** 2) * 1j * 50
    
# create a vector field A = x + y
# for row in A:
#     row += x ** 2
# for col in A.T:
#     col += 1j * y ** 2
    
p.quiver(x, y, np.real(A), np.imag(A))
p.xticks(x)
p.yticks(y)
p.grid(True)

L = op.laplacian(x, y, A)

print np.max(L), np.min(L)

# because of the numerical derivatives, points are shifted over by 1 dx
xd = x[1:-1] 
yd = y[1:-1] 

p.figure()
p.quiver(xd, yd, np.real(L), np.imag(L))
p.xticks(xd)
p.yticks(yd)
p.grid(True)

# now let's test the laplacian on a scalar field
A = np.real(A)

p.figure()
ex = (np.min(y), np.max(y), np.min(x), np.max(x))
implot = p.imshow(A, origin="lower", extent=ex)
implot.set_cmap("hot")
p.colorbar()
p.xticks(x)
p.yticks(y)

# calculate the laplacian again with the scalar field
L = op.laplacian(x, y, A)
print np.max(L), np.min(L)

# because of the numerical derivatives, points are shifted over by 1 dx
xd = x[1:-1] 
yd = y[1:-1] 

p.figure()
ex = (np.min(yd), np.max(yd), np.min(xd), np.max(xd))
implot = p.imshow(L, origin="lower", extent=ex)
implot.set_cmap("hot")
p.colorbar()
p.xticks(xd)
p.yticks(yd)

p.show()
