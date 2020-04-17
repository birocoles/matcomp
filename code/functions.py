import numpy as np

def dot_product(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^n.

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
