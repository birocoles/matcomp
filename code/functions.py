import numpy as np
from numba import jit

# dot product

def dot_product_dumb(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N.

    The code uses a simple for to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N real elements.

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
    x, y are elements of R^N.

    The code uses numpy.sum.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N real elements.

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
    x, y are elements of R^N.

    The code uses numba.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N real elements.

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


# Fourier Transform

def DFT_dumb(x, scale=None):
    '''
    Compute the Discrete Fourier Transform (DFT) of the 1D array x.

    The code is a slow version of the algorithm.

    Parameters
    ----------
    x : array 1D
        Vector with N elements.
    scale : {None, 'ortho'}, optional
        None (the default normalization) has the transform unscaled
        and requires the inverse transform to be scaled by 1/N.
        'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
        transform to be scaled by 1/sqrt{N}. In this case, the FT is unitary.

    Returns
    -------
    X : array 1D
        DFT of x.
    '''
    assert scale in [None, 'sqrtn'], "scale must be None or 'sqrtn'"
    x = np.asarray(x)
    # define the Fourier matrix
    N = x.size
    kn = np.outer(np.arange(N), np.arange(N))
    Fn = np.exp(-2j * np.pi * kn / N)
    # compute the DFT
    if scale is None:
        X = np.dot(Fn, x)
    else:
        X = np.dot(Fn, x)/np.sqrt(N)

    return X

def IDFT_dumb(x, scale='N'):
    '''
    Compute the Inverse Discrete Fourier Transform (IDFT) of the 1D array x.

    The code is a slow version of the algorithm.

    Parameters
    ----------
    x : array 1D
        Vector with N elements.
    scale : {None, 'ortho'}, optional
        'N' (the default normalization) has the transform unscaled
        and requires the inverse transform to be scaled by 1/N.
        'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
        transform to be scaled by 1/sqrt{N}. In this case, the FT is unitary.

    Returns
    -------
    X : array 1D
        DFT of x.
    '''
    assert scale in ['N', 'sqrtn'], "scale must be 'N' or 'sqrtn'"
    x = np.asarray(x)
    # define the Fourier matrix
    N = x.size
    kn = np.outer(np.arange(N), np.arange(N))
    Fn = np.exp(2j * np.pi * kn / N)
    # compute the DFT
    if scale is 'N':
        X = np.dot(Fn, x)/N
    else:
        X = np.dot(Fn, x)/np.sqrt(N)

    return X
