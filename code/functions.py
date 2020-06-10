import numpy as np
from numba import jit, prange

# dot product

def dot_real_dumb(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    if check_input is True:
        assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
        assert x.size == y.size, 'x and y must have the same size'

    result = 0
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result += x.real[i]*y.real[i]

    return result


def dot_real_numpy(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses numpy.sum.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    if check_input is True:
        assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
        assert x.size == y.size, 'x and y must have the same size'

    # the '.real' forces the code to use
    # only the real part of the arrays
    result = np.sum(x.real*y.real)

    return result


@jit(nopython=True)
def dot_real_numba(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses numba.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    if check_input is True:
        assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
        assert x.size == y.size, 'x and y must have the same size'

    result = 0
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result += x.real[i]*y.real[i]

    return result


@jit(nopython=True, parallel=True)
def dot_real_parallel(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses numba with parallelization.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    if check_input is True:
        assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
        assert x.size == y.size, 'x and y must have the same size'

    result = 0
    for i in prange(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result += x.real[i]*y.real[i]

    return result


def dot_complex_dumb(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    result_real = dot_real_dumb(x.real, y.real, check_input=False)
    result_real -= dot_real_dumb(x.imag, y.imag, check_input=False)
    result_imag = dot_real_dumb(x.real, y.imag, check_input=False)
    result_imag += dot_real_dumb(x.imag, y.real, check_input=False)
    result = result_real + 1j*result_imag

    return result


def dot_complex_numpy(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    The code uses numpy.sum.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    result_real = dot_real_numpy(x.real, y.real, check_input=False)
    result_real -= dot_real_numpy(x.imag, y.imag, check_input=False)
    result_imag = dot_real_numpy(x.real, y.imag, check_input=False)
    result_imag += dot_real_numpy(x.imag, y.real, check_input=False)
    result = result_real + 1j*result_imag

    return result


def dot_complex_numba(x, y):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    The code uses numba.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    result_real = dot_real_numba(x.real, y.real, check_input=False)
    result_real -= dot_real_numba(x.imag, y.imag, check_input=False)
    result_imag = dot_real_numba(x.real, y.imag, check_input=False)
    result_imag += dot_real_numba(x.imag, y.real, check_input=False)
    result = result_real + 1j*result_imag

    return result


def dot_complex(x, y, conjugate=False, function='numba'):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    conjugate : boolean
        If True, uses the complex conjugate of y. Default is False.

    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
    assert x.size == y.size, 'x and y must have the same size'

    dot_real = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'parallel': dot_real_parallel
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function))

    if conjugate is True:
        result_real = dot_real[function](x.real, y.real, check_input=False)
        result_real += dot_real[function](x.imag, y.imag, check_input=False)
        result_imag = dot_real[function](x.real, y.imag, check_input=False)
        result_imag -= dot_real[function](x.imag, y.real, check_input=False)
    else:
        result_real = dot_real[function](x.real, y.real, check_input=False)
        result_real -= dot_real[function](x.imag, y.imag, check_input=False)
        result_imag = dot_real[function](x.real, y.imag, check_input=False)
        result_imag += dot_real[function](x.imag, y.real, check_input=False)

    result = result_real + 1j*result_imag

    return result


# Hadamard (entrywise) product

def hadamard_real_dumb(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses a simple doubly nested loop to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    if check_input is True:
        assert x.shape == y.shape, 'x and y must have the same shape'
        assert (x.ndim == 1) or (x.ndim == 2), 'x and y must be vectors \
or matrices'

    result = np.empty_like(x)
    if x.ndim == 1:
        for i in range(x.shape[0]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] = x.real[i]*y.real[i]
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i,j] = x.real[i,j]*y.real[i,j]

    return result


def hadamard_real_numpy(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses the asterisk (star) operator.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    if check_input is True:
        assert x.shape == y.shape, 'x and y must have the same shape'
        assert (x.ndim == 1) or (x.ndim == 2), 'x and y must be vectors \
or matrices'

    # the '.real' forces the code to use
    # only the real part of the arrays
    result = x.real*y.real

    return result


@jit(nopython=True)
def hadamard_real_numba(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses numba.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    if check_input is True:
        assert x.shape == y.shape, 'x and y must have the same shape'
        assert (x.ndim == 1) or (x.ndim == 2), 'x and y must be vectors \
or matrices'

    result = np.empty_like(x)
    if x.ndim == 1:
        for i in range(x.shape[0]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] = x.real[i]*y.real[i]
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i,j] = x.real[i,j]*y.real[i,j]

    return result


@jit(nopython=True, parallel=True)
def hadamard_real_parallel(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses numba with automatic parallelization.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    if check_input is True:
        assert x.shape == y.shape, 'x and y must have the same shape'
        assert (x.ndim == 1) or (x.ndim == 2), 'x and y must be vectors \
or matrices'

    result = np.empty_like(x)
    if x.ndim == 1:
        for i in prange(x.shape[0]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] = x.real[i]*y.real[i]
    else:
        for i in prange(x.shape[0]):
            for j in range(x.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i,j] = x.real[i,j]*y.real[i,j]

    return result


def hadamard_complex(x, y, function='numba'):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors or matrices having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex vectors or matrices having the same shape.

    function : string
        Function to be used for computing the real Hadamard product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    assert x.shape == y.shape, 'x and y must have the same shape'
    assert (x.ndim == 1) or (x.ndim == 2), 'x and y must be vectors or matrices'

    hadamard_real = {
        'dumb': hadamard_real_dumb,
        'numpy': hadamard_real_numpy,
        'numba': hadamard_real_numba,
        'parallel': hadamard_real_parallel
    }
    if function not in hadamard_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = hadamard_real[function](x.real, y.real, check_input=False)
    result_real -= hadamard_real[function](x.imag, y.imag, check_input=False)
    result_imag = hadamard_real[function](x.real, y.imag, check_input=False)
    result_imag += hadamard_real[function](x.imag, y.real, check_input=False)

    result = result_real + 1j*result_imag

    return result


# Outer product

# def outer_real_dumb(x, y, check_input=True):
#     '''
#     Compute the dot product of x and y, where
#     x, y are elements of R^N. The imaginary parts are ignored.
#
#     The code uses a simple "for" to iterate on the arrays.
#
#     Parameters
#     ----------
#     x, y : arrays 1D
#         Vectors with N elements.
#
#     check_input : boolean
#         If True, verify if the input is valid. Default is True.
#
#     Returns
#     -------
#     result : scalar
#         Dot product of x and y.
#     '''
#     if check_input is True:
#         assert x.ndim == y.ndim == 1, 'x and y must be 1D arrays'
#         assert x.size == y.size, 'x and y must have the same size'
#
#     result = 0
#     for i in range(x.size):
#         # the '.real' forces the code to use
#         # only the real part of the arrays
#         result += x.real[i]*y.real[i]
#
#     return result


# Fourier Transform

# def DFT_dumb(x, scale=None):
#     '''
#     Compute the Discrete Fourier Transform (DFT) of the 1D array x.
#
#     The code is a slow version of the algorithm.
#
#     Parameters
#     ----------
#     x : array 1D
#         Vector with N elements.
#     scale : {None, 'ortho'}, optional
#         None (the default normalization) has the transform unscaled
#         and requires the inverse transform to be scaled by 1/N.
#         'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
#         transform to be scaled by 1/sqrt{N}. In this case, the FT is unitary.
#
#     Returns
#     -------
#     X : array 1D
#         DFT of x.
#     '''
#     assert scale in [None, 'sqrtn'], "scale must be None or 'sqrtn'"
#     x = np.asarray(x)
#     # define the Fourier matrix
#     N = x.size
#     kn = np.outer(np.arange(N), np.arange(N))
#     Fn = np.exp(-2j * np.pi * kn / N)
#     # compute the DFT
#     if scale is None:
#         X = np.dot(Fn, x)
#     else:
#         X = np.dot(Fn, x)/np.sqrt(N)
#
#     return X
#
# def IDFT_dumb(x, scale='N'):
#     '''
#     Compute the Inverse Discrete Fourier Transform (IDFT) of the 1D array x.
#
#     The code is a slow version of the algorithm.
#
#     Parameters
#     ----------
#     x : array 1D
#         Vector with N elements.
#     scale : {None, 'ortho'}, optional
#         'N' (the default normalization) has the transform unscaled
#         and requires the inverse transform to be scaled by 1/N.
#         'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
#         transform to be scaled by 1/sqrt{N}. In this case, the FT is unitary.
#
#     Returns
#     -------
#     X : array 1D
#         DFT of x.
#     '''
#     assert scale in ['N', 'sqrtn'], "scale must be 'N' or 'sqrtn'"
#     x = np.asarray(x)
#     # define the Fourier matrix
#     N = x.size
#     kn = np.outer(np.arange(N), np.arange(N))
#     Fn = np.exp(2j * np.pi * kn / N)
#     # compute the DFT
#     if scale is 'N':
#         X = np.dot(Fn, x)/N
#     else:
#         X = np.dot(Fn, x)/np.sqrt(N)
#
#     return X
