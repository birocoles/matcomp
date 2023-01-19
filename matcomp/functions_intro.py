import numpy as np
from numba import njit
from scipy.linalg import dft


# scalar-vector product


def scalar_vec_real_dumb(a, x, check_input=True):
    """
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the array.

    Parameters
    ----------
    a : scalar
        Real number.

    x : array 1D
        Vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Product of a and x.
    """
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, "a must be a scalar"
        assert x.ndim == 1, "x must be a 1D array"

    result = np.empty_like(x, dtype="float")
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result[i] = a.real * x.real[i]

    return result


def scalar_vec_real_numpy(a, x, check_input=True):
    """
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.

    The code uses numpy.

    Parameters
    ----------
    a : scalar
        Real number.

    x : array 1D
        Vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Product of a and x.
    """
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, "a must be a scalar"
        assert x.ndim == 1, "x must be a 1D array"

    result = a.real * x.real

    return result


@njit
def scalar_vec_real_numba(a, x, check_input=True):
    """
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.

    The code uses numba.

    Parameters
    ----------
    a : scalar
        Real number.

    x : array 1D
        Vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Product of a and x.
    """
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, "a must be a scalar"
        assert x.ndim == 1, "x must be a 1D array"

    result = np.empty_like(x, dtype="float")
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result[i] = a.real * x.real[i]

    return result


def scalar_vec_complex(a, x, check_input=True, function="numba"):
    """
    Compute the dot product of a is a complex number and x
    is a complex vector.

    Parameters
    ----------
    a : scalar
        Complex number.

    x : array 1D
        Complex vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real scalar-vector product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : scalar
        Product of a and x.
    """
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, "a must be a scalar"
        assert x.ndim == 1, "x must be a 1D array"

    scalar_vec_real = {
        "dumb": scalar_vec_real_dumb,
        "numpy": scalar_vec_real_numpy,
        "numba": scalar_vec_real_numba,
    }
    if function not in scalar_vec_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = scalar_vec_real[function](a.real, x.real, check_input=False)
    result_real -= scalar_vec_real[function](a.imag, x.imag, check_input=False)
    result_imag = scalar_vec_real[function](a.real, x.imag, check_input=False)
    result_imag += scalar_vec_real[function](a.imag, x.real, check_input=False)

    result = result_real + 1j * result_imag

    return result


# dot product


def dot_real_dumb(x, y, check_input=True):
    """
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
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.size == y.size, "x and y must have the same size"

    result = 0
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result += x.real[i] * y.real[i]

    return result


def dot_real_numpy(x, y, check_input=True):
    """
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
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.size == y.size, "x and y must have the same size"

    # the '.real' forces the code to use
    # only the real part of the arrays
    result = np.sum(x.real * y.real)

    return result


# @jit(nopython=True)
@njit
def dot_real_numba(x, y, check_input=True):
    """
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses numba jit.

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
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.size == y.size, "x and y must have the same size"

    result = 0
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result += x.real[i] * y.real[i]

    return result


def dot_complex_dumb(x, y, check_input=True):
    """
    Compute the dot product of x and y, where
    x, y are elements of C^N.

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
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.size == y.size, "x and y must have the same size"

    result_real = dot_real_dumb(x.real, y.real, check_input=False)
    result_real -= dot_real_dumb(x.imag, y.imag, check_input=False)
    result_imag = dot_real_dumb(x.real, y.imag, check_input=False)
    result_imag += dot_real_dumb(x.imag, y.real, check_input=False)
    result = result_real + 1j * result_imag

    return result


def dot_complex_numpy(x, y):
    """
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
    """
    assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
    assert x.size == y.size, "x and y must have the same size"

    result_real = dot_real_numpy(x.real, y.real, check_input=False)
    result_real -= dot_real_numpy(x.imag, y.imag, check_input=False)
    result_imag = dot_real_numpy(x.real, y.imag, check_input=False)
    result_imag += dot_real_numpy(x.imag, y.real, check_input=False)
    result = result_real + 1j * result_imag

    return result


def dot_complex_numba(x, y):
    """
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
    """
    assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
    assert x.size == y.size, "x and y must have the same size"

    result_real = dot_real_numba(x.real, y.real, check_input=False)
    result_real -= dot_real_numba(x.imag, y.imag, check_input=False)
    result_imag = dot_real_numba(x.real, y.imag, check_input=False)
    result_imag += dot_real_numba(x.imag, y.real, check_input=False)
    result = result_real + 1j * result_imag

    return result


def dot_complex(x, y, conjugate=False, check_input=True, function="numba"):
    """
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    conjugate : boolean
        If True, uses the complex conjugate of x. Default is False.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.size == y.size, "x and y must have the same size"

    dot_real = {
        "dumb": dot_real_dumb,
        "numpy": dot_real_numpy,
        "numba": dot_real_numba,
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

    result = result_real + 1j * result_imag

    return result


# Hadamard (entrywise) product


def hadamard_real_dumb(x, y, check_input=True):
    """
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
    """
    if check_input is True:
        assert x.shape == y.shape, "x and y must have the same shape"
        assert (x.ndim == 1) or (
            x.ndim == 2
        ), "x and y must be vectors or matrices"

    result = np.empty_like(x)
    if x.ndim == 1:
        for i in range(x.shape[0]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] = x.real[i] * y.real[i]
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i, j] = x.real[i, j] * y.real[i, j]

    return result


def hadamard_real_numpy(x, y, check_input=True):
    """
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
    """
    if check_input is True:
        assert x.shape == y.shape, "x and y must have the same shape"
        assert (x.ndim == 1) or (
            x.ndim == 2
        ), "x and y must be vectors or matrices"

    # the '.real' forces the code to use
    # only the real part of the arrays
    result = x.real * y.real

    return result


# @jit(nopython=True)
@njit
def hadamard_real_numba(x, y, check_input=True):
    """
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
    """
    if check_input is True:
        assert x.shape == y.shape, "x and y must have the same shape"
        assert (x.ndim == 1) or (
            x.ndim == 2
        ), "x and y must be vectors or matrices"

    result = np.empty_like(x)
    if x.ndim == 1:
        for i in range(x.shape[0]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] = x.real[i] * y.real[i]
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i, j] = x.real[i, j] * y.real[i, j]

    return result


def hadamard_complex(x, y, check_input=True, function="numba"):
    """
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors or matrices having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real Hadamard product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    """
    if check_input is True:
        assert x.shape == y.shape, "x and y must have the same shape"
        assert (x.ndim == 1) or (
            x.ndim == 2
        ), "x and y must be vectors or matrices"

    hadamard_real = {
        "dumb": hadamard_real_dumb,
        "numpy": hadamard_real_numpy,
        "numba": hadamard_real_numba,
    }
    if function not in hadamard_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = hadamard_real[function](x.real, y.real, check_input=False)
    result_real -= hadamard_real[function](x.imag, y.imag, check_input=False)
    result_imag = hadamard_real[function](x.real, y.imag, check_input=False)
    result_imag += hadamard_real[function](x.imag, y.real, check_input=False)

    result = result_real + 1j * result_imag

    return result


# Outer product


def outer_real_dumb(x, y, check_input=True):
    """
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"

    # result = np.zeros((x.size, y.size))
    result = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i, j] = x.real[i] * y.real[j]

    return result


def outer_real_numpy(x, y, check_input=True):
    """
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses numpy.newaxis for broadcasting
    https://numpy.org/devdocs/user/theory.broadcasting.html

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"

    # the '.real' forces the code to use
    # only the real part of the arrays
    result = x.real[:, np.newaxis] * y.real[np.newaxis, :]

    return result


# @jit(nopython=True)
@njit
def outer_real_numba(x, y, check_input=True):
    """
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses numba.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"

    result = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i, j] = x.real[i] * y.real[j]

    return result


def outer_complex(x, y, check_input=True, function="numba"):
    """
    Compute the outer product of x and y, where x and y are complex vectors.

    Parameters
    ----------
    x, y : 1D arrays
        Complex vectors.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real outer product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : 2D array
        Outer product of x and y.
    """
    if check_input is True:
        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"

    outer_real = {
        "dumb": outer_real_dumb,
        "numpy": outer_real_numpy,
        "numba": outer_real_numba,
    }
    if function not in outer_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = outer_real[function](x.real, y.real, check_input=False)
    result_real -= outer_real[function](x.imag, y.imag, check_input=False)
    result_imag = outer_real[function](x.real, y.imag, check_input=False)
    result_imag += outer_real[function](x.imag, y.real, check_input=False)

    result = result_real + 1j * result_imag

    return result


# matrix-vector product


def matvec_real_dumb(A, x, check_input=True):
    """
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code uses a simple doubly nested "for" to iterate on the arrays.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    """
    if check_input is True:
        assert (A.ndim == 2) and (
            x.ndim == 1
        ), "A and x must be 2D and 1D arrays, respectively"
        assert A.shape[1] == x.size, "A and x do not match"

    result = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] += A.real[i, j] * x.real[j]

    return result


@njit
def matvec_real_numba(A, x, check_input=True):
    """
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code uses numba jit.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    """
    if check_input is True:
        assert (A.ndim == 2) and (
            x.ndim == 1
        ), "A and x must be 2D and 1D arrays, respectively"
        assert A.shape[1] == x.size, "A and x do not match"

    result = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # the '.real' forces the code to use
            # only the real part of the arrays
            result[i] += A.real[i, j] * x.real[j]

    return result


def matvec_real_dot(A, x, check_input=True, function="numba"):
    """
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a dot product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : array 1D
        Product of A and x.
    """
    if check_input is True:
        assert (A.ndim == 2) and (
            x.ndim == 1
        ), "A and x must be 2D and 1D arrays, respectively"
        assert A.shape[1] == x.size, "A and x do not match"

    dot_real = {
        "dumb": dot_real_dumb,
        "numpy": dot_real_numpy,
        "numba": dot_real_numba,
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        result[i] = dot_real[function](A[i, :], x, check_input=False)

    return result


def matvec_real_columns(A, x, check_input=True, function="numba"):
    """
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a scalar-vector product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : array 1D
        Product of A and x.
    """
    if check_input is True:
        assert (A.ndim == 2) and (
            x.ndim == 1
        ), "A and x must be 2D and 1D arrays, respectively"
        assert A.shape[1] == x.size, "A and x do not match"

    scalar_vec_real = {
        "dumb": scalar_vec_real_dumb,
        "numpy": scalar_vec_real_numpy,
        "numba": scalar_vec_real_numba,
    }
    if function not in scalar_vec_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        result += scalar_vec_real[function](x[j], A[:, j], check_input=False)

    return result


def matvec_complex(A, x, check_input=True, function="numba"):
    """
    Compute the matrix-vector product of an NxM matrix A and
    a Mx1 vector x.

    Parameters
    ----------
    A : array 2D
        NxM matrix.

    x : array 1D
        Mx1 vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real mattrix-vectorvec product.
        The function name must be 'dumb', 'numba', 'dot' or 'columns'.
        Default is 'numba'.

    Returns
    -------
    result : array 1D
        Product of A and x.
    """
    if check_input is True:
        assert (A.ndim == 2) and (
            x.ndim == 1
        ), "A and x must be 2D and 1D arrays, respectively"
        assert A.shape[1] == x.size, "A and x do not match"

    matvec_real = {
        "dumb": matvec_real_dumb,
        "numba": matvec_real_numba,
        "dot": matvec_real_dot,
        "columns": matvec_real_columns,
    }
    if function not in matvec_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = matvec_real[function](A.real, x.real, check_input=False)
    result_real -= matvec_real[function](A.imag, x.imag, check_input=False)
    result_imag = matvec_real[function](A.real, x.imag, check_input=False)
    result_imag += matvec_real[function](A.imag, x.real, check_input=False)

    result = result_real + 1j * result_imag

    return result


# matrix-matrix product


def matmat_real_dumb(A, B, check_input=True):
    """
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code uses a simple triply nested "for" to iterate on the arrays.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i, j] += A.real[i, k] * B.real[k, j]

    return result


@njit
def matmat_real_numba(A, B, check_input=True):
    """
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code uses numba.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                # the '.real' forces the code to use
                # only the real part of the arrays
                result[i, j] += A.real[i, k] * B.real[k, j]

    return result


def matmat_real_dot(A, B, check_input=True, function="numba"):
    """
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces one "for" by a dot product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    dot_real = {
        "dumb": dot_real_dumb,
        "numpy": dot_real_numpy,
        "numba": dot_real_numba,
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.empty((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = dot_real[function](
                A[i, :], B[:, j], check_input=False
            )

    return result


def matmat_real_columns(A, B, check_input=True, function="numba"):
    """
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces one "for" by a scalar-vector product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real scalar-vector product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    scalar_vec_real = {
        "dumb": scalar_vec_real_dumb,
        "numpy": scalar_vec_real_numpy,
        "numba": scalar_vec_real_numba,
    }
    if function not in scalar_vec_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.zeros((A.shape[0], B.shape[1]))
    for j in range(B.shape[1]):
        for k in range(A.shape[1]):
            result[:, j] += scalar_vec_real[function](
                B[k, j], A[:, k], check_input=False
            )

    return result


def matmat_real_matvec(A, B, check_input=True, function="numba"):
    """
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "for" by a matrix-vector product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real matrix-vector product.
        The function name must be 'dumb', 'numba', 'dot' and 'columns'.
        Default is 'numba'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    matvec_real = {
        "dumb": matvec_real_dumb,
        "numba": matvec_real_numba,
        "dot": matvec_real_dot,
        "columns": matvec_real_columns,
    }
    if function not in matvec_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.empty((A.shape[0], B.shape[1]))
    for j in range(B.shape[1]):
        result[:, j] = matvec_real[function](A, B[:, j], check_input=False)

    return result


def matmat_real_outer(A, B, check_input=True, function="numba"):
    """
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "for" by an outer product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real outer product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    outer_real = {
        "dumb": outer_real_dumb,
        "numpy": outer_real_numpy,
        "numba": outer_real_numba,
    }
    if function not in outer_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.zeros((A.shape[0], B.shape[1]))
    for k in range(A.shape[1]):
        result += outer_real[function](A[:, k], B[k, :], check_input=False)

    return result


def matmat_complex(A, B, check_input=True, function="numba"):
    """
    Compute the matrix-matrix product of A and B, where
    A in C^NxM and B in C^MxP.

    Parameters
    ----------
    A, B : 2D arrays
        Complex matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real and imaginary parts.
        The function name must be 'dumb', 'numba', 'dot', 'columns', 'matvec'
        and 'outer'. Default is 'numba'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    """
    if check_input is True:
        assert A.ndim == B.ndim == 2, "A and B must be 2D arrays"
        assert A.shape[1] == B.shape[0], "A and B do not match"

    matmat_real = {
        "dumb": matmat_real_dumb,
        "numba": matmat_real_numba,
        "dot": matmat_real_dot,
        "columns": matmat_real_columns,
        "matvec": matmat_real_matvec,
        "outer": matmat_real_outer,
    }
    if function not in matmat_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = matmat_real[function](A.real, B.real, check_input=False)
    result_real -= matmat_real[function](A.imag, B.imag, check_input=False)
    result_imag = matmat_real[function](A.real, B.imag, check_input=False)
    result_imag += matmat_real[function](A.imag, B.real, check_input=False)

    result = result_real + 1j * result_imag

    return result


# triangular matrix-vector product


def matvec_triu_real_dot(A, x, check_input=True, function="numba"):
    """
    Compute the matrix-vector product of A and x, where
    A is an upper triangular matrix in R^NxM and x in R^M.
    The imaginary parts are ignored.

    The code replaces a for by a dot product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numba'.

    Returns
    -------
    result : array 1D
        Product of A and x.
    """
    if check_input is True:
        assert (A.ndim == 2) and (
            x.ndim == 1
        ), "A and x must be 2D and 1D arrays, respectively"
        assert A.shape[1] == x.size, "A and x do not match"

    dot_real = {
        "dumb": dot_real_dumb,
        "numpy": dot_real_numpy,
        "numba": dot_real_numba,
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function))

    result = np.zeros(A.shape[0])
    for i in range(A.shape[0] - 1):
        result[i] = dot_real[function](A[i, i:], x[i:], check_input=False)
    result[-1] = A[-1, -1] * x[-1]

    return result
