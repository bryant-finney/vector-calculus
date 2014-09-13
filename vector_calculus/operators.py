'''
    Author: Bryant Finney
    Date: 7 September 2014
'''

import numpy as np

def gradient(x, y, A):
    if len(np.shape(x)) is not 1:
        raise ValueError("x values should be a 1 dimensional array.")
    if len(np.shape(y)) is not 1:
        raise ValueError("y values should be a 1 dimensional array.")
    if np.shape(A) != (len(y), len(x)):
        raise ValueError("A should be a 2D array of size len(y) by len(x).")
    # preallocate output array
    g = np.zeros((len(y) - 1, len(x) - 1), dtype=np.complex128)
    # loop through rows and take the partial derivative with respect to x
    for i, row in enumerate(A):
        if i is len(y) - 1:
            break
        g[i, :] += np.diff(row) / np.diff(x)
        
    # loop through columns and take the partial derivative with respect to y
    for i, col in enumerate(A.T):
        if i is len(x) - 1:
            break
        g[:, i] += np.diff(col) / np.diff(y) * 1j
        
    return g

def divergence(x, y, A):
    if len(np.shape(x)) is not 1:
        raise ValueError("x values should be a 1 dimensional array.")
    if len(np.shape(y)) is not 1:
        raise ValueError("y values should be a 1 dimensional array.")
    if np.shape(A) != (len(y), len(x)):
        raise ValueError("A should be a 2D array of size len(y) by len(x).")
    # preallocate output array
    d = np.zeros((len(y) - 1, len(x) - 1))
    # loop through rows and take the partial derivative of the y component 
    # with respect to x
    for i, row in enumerate(A):
        if i is len(y) - 1:
            break
        d[i, :] += np.diff(np.real(row)) / np.diff(x)
    # loop through columns and take the partial derivative of the x component 
    # with respect to y
    for i, col in enumerate(A.T):
        if i is len(x) - 1:
            break
        d[:, i] += np.diff(np.imag(col)) / np.diff(y)

    return d
    
def curl(x, y, A):
    if len(np.shape(x)) is not 1:
        raise ValueError("x values should be a 1 dimensional array.")
    if len(np.shape(y)) is not 1:
        raise ValueError("y values should be a 1 dimensional array.")
    if np.shape(A) != (len(y), len(x)):
        raise ValueError("A should be a 2D array of size len(y) by len(x).")
    # preallocate output array
    c = np.zeros((len(y) - 1, len(x) - 1))
    # loop through rows and take the partial derivative of the y component 
    # with respect to x
    for i, row in enumerate(A):
        if i is len(y) - 1:
            break
        c[i, :] += np.diff(np.imag(row)) / np.diff(x)
    # loop through columns and take the partial derivative of the x component 
    # with respect to y
    for i, col in enumerate(A.T):
        if i is len(x) - 1:
            break
        c[:, i] -= np.diff(np.real(col)) / np.diff(y)

    return c

def laplacian(x, y, A):
    if len(np.shape(x)) is not 1:
        raise ValueError("x values should be a 1 dimensional array.")
    if len(np.shape(y)) is not 1:
        raise ValueError("y values should be a 1 dimensional array.")
    if np.shape(A) != (len(y), len(x)):
        raise ValueError("A should be a 2D array of size len(y) by len(x).")
    
    # determine if the input is a vector field or a scalar field. Each is
    # handled differently
    if A.dtype == np.complex128 or A.dtype == np.complex64:
        # we have a vector field; preallocate vector output array
        L = laplacian(x, y, np.real(A)) + laplacian(x, y, np.imag(A)) * 1j
    else:
        # we have a scalar field; preallocate scalar output array
        L = np.zeros((len(y) - 2, len(x) - 2))
        for i, row in enumerate(A):
            if i is len(y) - 2:
                break
            # take the second derivative with respect to x
            L[i, :] += np.diff(np.diff(row) / np.diff(x)) / np.diff(x[:-1] + 0.5 * np.diff(x))
        for i, col in enumerate(A.T):
            if i is len(x) - 2:
                break
            # take the second derivative with respect to y
            L[:, i] += np.diff(np.diff(col) / np.diff(y)) / np.diff(y[:-1] + 0.5 * np.diff(y))
    
    return L
    
