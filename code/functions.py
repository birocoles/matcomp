import numpy as np
from numba import jit


def dot_product_dumb(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^n.

    The code uses a simple for to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with n real elements.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    result = 0
    for i in range(x.size):
        result += x[i]*y[i]

    return result


def dot_product_numpy(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^n.

    The code uses numpy.sum.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with n real elements.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    result = np.sum(x*y)
    return result


@jit(nopython=True)
def dot_product_numba(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^n.

    The code uses numba for speed up computations.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with n real elements.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    result = 0
    for i in range(x.size):
        result += x[i]*y[i]
    return result
