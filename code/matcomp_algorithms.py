import numpy as np

# Fourier Transform

def DFT_matrix(N, scale=None, conjugate=False, check_input=True):
    '''
    Compute the Discrete Fourier Transform (DFT) matrix.

    Parameters
    ----------
    N : int
        Order of DFT matrix.

    scale : None, 'n' or 'sqrtn'
        None (the default normalization) has the transform unscaled
        and requires the inverse transform to be scaled by 1/N.
        'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
        transform to be scaled by 1/sqrt{N}. In this case, the transform
        is unitary.

    conjugate : boolean
        If True, returns the complex conjugate of FN. Default is False.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Fn : 2D array
        DFT matrix.
    '''
    if check_input is True:
        assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"
        assert isinstance(N, int) and N > 0, 'N must be a positive integer'

    kn = np.outer(a=np.arange(N), b=np.arange(N))

    # define the unscaled Fourier matrix
    if conjugate is True:
        Fn = np.exp( 2 * np.pi * 1j * kn / N)
    else:
        Fn = np.exp(-2 * np.pi * 1j * kn / N)

    if scale is 'n':
        return Fn/N
    elif scale is 'sqrtn':
        return Fn/np.sqrt(N)
    else:
        return Fn


def DFT_dumb(x, scale=None):
    '''
    Compute the Discrete Fourier Transform (DFT) of the 1D array x.

    The code is a slow version of the algorithm.

    Parameters
    ----------
    x : array 1D
        Vector with N elements.

    scale : None, 'n' or 'sqrtn'
        None (the default normalization) has the transform unscaled
        and requires the inverse transform to be scaled by 1/N.
        'n' scales the transform by 1/N.
        'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
        transform to be scaled by 1/sqrt{N}. In this case, the transform
        is unitary.

    Returns
    -------
    X : array 1D
        DFT of x.
    '''
    assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"

    x = np.asarray(x)

    # define the Fourier matrix
    N = x.size
    Fn = DFT_matrix(N=N, scale=scale, conjugate=False, check_input=False)

    # compute the DFT
    X = np.dot(a=Fn, b=x)

    return X


def IDFT_dumb(X, scale='n'):
    '''
    Compute the Inverse Discrete Fourier Transform (IDFT) of the 1D array x.

    The code is a slow version of the algorithm.

    Parameters
    ----------
    X : array 1D
        Complex vector with N elements.

    scale : None, 'n' or 'sqrtn'
        None has the inverse transform unscaled.
        'n' scales the transform by 1/N (default normalization).
        'sqrtn' scales the inverse transform by 1/sqrt{N} and requires the
        transform to be scaled by 1/sqrt{N}. In this case, the transform
        is unitary.

    Returns
    -------
    x : array 1D
        IDFT of X.
    '''
    assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"

    X = np.asarray(X)

    # define the Fourier matrix
    N = X.size
    Fn_conj = DFT_matrix(N=N, scale=scale, conjugate=True, check_input=False)

    # compute the DFT
    x = np.dot(a=Fn_conj, b=X)

    return x


def DFT_recursive(x, scale=None):
    '''
    Compute the Discrete Fourier Transform (DFT) of the 1D array x.

    The code uses the recursive strategy of Cooley-Tukey (Golub and Van Loan,
    2013, Algorithm 1.4.1, p. 35). The number of data must be a power of 2.

    Parameters
    ----------
    x : array 1D
        Vector with N elements.

    scale : None, 'n' or 'sqrtn'
        None (the default normalization) has the transform unscaled
        and requires the inverse transform to be scaled by 1/N.
        'n' scales the transform by 1/N.
        'sqrtn' scales the transform by 1/sqrt{N} and requires the inverse
        transform to be scaled by 1/sqrt{N}. In this case, the transform
        is unitary.

    Returns
    -------
    X : array 1D
        DFT of x.
    '''
    assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"

    x = np.asarray(x)

    N = x.size

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")

    X = _FT_recursive(data=x, conjugate=False)

    if scale is 'n':
        return X/N
    elif scale is 'sqrtn':
        return X/np.sqrt(N)
    else:
        return X


def IDFT_recursive(X, scale='n'):
    '''
    Compute the Inverse Discrete Fourier Transform (IDFT) of the 1D array x.

    The code uses the recursive strategy of Cooley-Tukey (Golub and Van Loan,
    2013, Algorithm 1.4.1, p. 35). The number of data must be a power of 2.

    Parameters
    ----------
    X : array 1D
        Complex vector with N elements.

    scale : None, 'n' or 'sqrtn'
        None has the inverse transform unscaled.
        'n' scales the transform by 1/N (default normalization).
        'sqrtn' scales the inverse transform by 1/sqrt{N} and requires the
        transform to be scaled by 1/sqrt{N}. In this case, the transform
        is unitary.

    Returns
    -------
    x : array 1D
        IDFT of X.
    '''
    assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"

    X = np.asarray(X)

    N = X.size

    if N % 2 > 0:
        raise ValueError("size of X must be a power of 2")

    x = _FT_recursive(data=X, conjugate=True)

    if scale is 'n':
        return x/N
    elif scale is 'sqrtn':
        return x/np.sqrt(N)
    else:
        return x


def _FT_recursive(data, conjugate=False):
    '''
    Compute the unscaled DFT/IDFT by using the recursive strategy of
    Cooley-Tukey (Golub and Van Loan, 2013, Algorithm 1.4.1, p. 35). The number
    of data must be a power of 2.

    Parameters
    ----------
    data : array 1D
        Vector with N elements.

    conjugate : boolean
        If True, returns the complex conjugate of the Fourier matrix.
        Default is False.

    Returns
    -------
    transformed_data : array 1D
        DFT/IDFT of data.
    '''

    N = data.size

    if conjugate:
        signal =  1
    else:
        signal = -1

    if N <= 64:
        # Define the Fourier matrix
        Fn = DFT_matrix(N=N, scale=None, conjugate=conjugate, check_input=True)
        # compute the DFT
        y = np.dot(a=Fn, b=data)
        return y
    else:
        y_even = _FT_recursive(data=data[0::2], conjugate=conjugate)
        y_odd  = _FT_recursive(data=data[1::2], conjugate=conjugate)
        omega = np.exp(signal * 1j * 2 * np.pi * np.arange(N//2)/N)
        return np.hstack([y_even + omega * y_odd,
                          y_even - omega * y_odd])


# Householder transformation
def House_vector(x, check_input=True):
    '''
    Compute the real Householder vector (Golub and Van Loan, 2013,
    Algorithm 5.1.1, p. 236) v, with v[0] = 1, such that matrix
    P = I - beta outer(v,v) (Householder reflection) is orthogonal
    and Px = norm_2(x) e_1.

    Parameters
    ----------
    x : array 1D
        Vector perpendicular to the Householder vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    v : array 1D
        Householder vector.

    beta : float
        Scalar equal to 1/dot(v,v).
    '''
    x = np.asarray(x)
    if check_input is True:
        assert x.ndim == 1, 'x must be a vector'
        #assert x.size > 1, 'x size must be greater than 1'

    N = x.size
    sigma = np.dot(x[1:], x[1:])
    v = np.hstack([1, x[1:]])

    if (sigma == 0) and (x[0] >= 0):
        beta = 0
    elif (sigma == 0) and (x[0] < 0):
        beta = -2
    else:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma/(x[0] + mu)
        beta = 2*v[0]**2/(sigma + v[0]**2)
        v /= v[0]

    return v, beta


def House_matvec(A, v, beta, order='AP', check_input=True):
        '''
        Compute the matrix-matrix product AP or PA, where P
        is a Householder matrix P = I - beta outer(v,v)
        (Golub and Van Loan, 2013, p. 236).

        Parameters
        ----------
        A : array 2D
            Matrix used to compute the product.

        v : array 1D
            Householder vector.

        beta : scalar
            Parameter 2/dot(v,v).

        order : string
            If 'PA', it defines the product AP. If 'AP',
            it defines the product PA. Default is 'AP'.

        check_input : boolean
            If True, verify if the input is valid. Default is True.

        Returns
        -------
        C : array 2D
            Matrix-matrix product.
        '''
        A = np.asarray(A)
        v = np.asarray(v)
        if check_input is True:
            assert A.ndim == 2, 'A must be a matrix'
            assert v.ndim == 1, 'v must be a vector'
            assert np.isscalar(beta), 'beta must be a scalar'
            assert order in ['PA', 'AP'], "order must be 'PA' or 'AP'"

        if order is 'AP':
            assert A.shape[1] == v.size, 'A shape[1] must be equal to v size'
            C = A - np.outer(np.dot(A, v), beta*v)
        else:
            assert v.size == A.shape[0], 'v size must be equal to A shape[0]'
            C = A - np.outer(beta*v, np.dot(v, A))

        return C

# Givens rotations
def Givens_rotation(a, b, check_input=True):
    '''
    Given real numbers a and b, this function computes c = cos(theta)
    and s = sin(theta) so that ca - sb = r and sa + cb = 0 (Golub and
    Van Loan, 2013, Algorithm 5.1.3, p. 240).

    Parameters
    ----------
    a, b : scalars
        They form the vector to be rotated.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    c, s : scalars
        Cosine and Sine of theta forming the Givens rotation matrix.
    '''
    if check_input is True:
        assert np.isscalar(a) and np.isscalar(b), 'a and b must be scalars'

    if b == 0:
        c = 1
        s = 0
    else:
        if np.abs(b) > np.abs(a):
            tau = -a/b
            s = 1/np.sqrt(1 + tau**2)
            c = s*tau
        else:
            tau = -b/a
            c = 1/np.sqrt(1 + tau**2)
            s = c*tau

    return c, s


def Givens_matvec(A, c, s, i, k, order='AG', check_input=True):
    '''
    Compute the matrix-matrix product AG or GTA, where G
    is a Givens rotation G(i, k, theta) (Golub and Van Loan, 2013,
    p. 241).

    Parameters
    ----------
    A : array 2D
        Matrix used to compute the product.

    c, s : scalars
        Cosine and Sine of theta forming the Givens rotation matrix.

    i, k : ints
        Indices of the Givens rotation matrix.

    order : string
        If 'AG', it defines the product AG. If 'GTA',
        it defines the product GTA. Default is 'AG'.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    C : array 2D
        Matrix-matrix product.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert np.isscalar(c) and np.isscalar(s), 'c and s must be scalars'
        assert isinstance(i, int) and (i >= 0), 'i must be a an integer >= 0'
        assert isinstance(k, int) and (k >= 0), 'k must be a an integer >= 0'
        assert (c**2 + s**2) == 1, 'c**2 + s**2 must be equal to 1'
        assert order in ['AG', 'GTA'], "order must be 'AG' or 'GTA'"

    M = A.shape[0]
    N = A.shape[1]
    G = np.array([[ c, s],
                  [-s, c]])
    if order is 'AG':
        assert (i < N) and (k < N), 'i and k must be < N'
        C = np.dot(A[:,[i,k]], G)
    else:
        assert (i < M) and (k < M), 'i and k must be < M'
        C = np.dot(G.T, A[[i,k],:])

    return C


def cs2rho(c, s, check_input=True):
    '''
    Compute a single float rho representing the matrix Z
    | c   s |
    |-s   c |
    according to Golub and Van Loan, 2013, p. 242.

    Parameters
    ----------
    c, s : scalars
        Scalars computed with function Givens_rotation.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    rho : float
        Representing scalar.
    '''
    if check_input is True:
        assert np.isscalar(c) and np.isscalar(s), 'c and s must be scalars'
        assert (c**2 + s**2) == 1, 'c**2 + s**2 must be equal to 1'

    if c == 0:
        rho = 1
    elif np.abs(s) < np.abs(c):
        rho = np.sign(c)*s/2
    else:
        rho = 2*np.sign(s)/c

    return rho


def rho2cs(rho, check_input=True):
    '''
    Compute c and s from the representing scalar rho
    obtained with function cs2rho (Golub and Van Loan, 2013, p. 242).

    Parameters
    ----------
    rho : scalar
        Representing scalar obtained with function cs2rho.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    c, s : float
        Scalars computed with function Givens_rotation.
    '''
    if check_input is True:
        assert np.isscalar(rho), 'rho must be a scalar'

    if rho == 1:
        c = 0
        s = 1
    elif np.abs(rho) < 1:
        s = 2*rho
        c = np.sqrt(1 - s**2)
    else:
        c = 2/rho
        s = np.sqrt(1 - c**2)

    return c, s


# QR factorization
def House_QR(A, check_input=True):
    '''
    Compute the Householder matrices H0, ..., HN-1 such that
    Q = H0 H1 ... HN-1 is orthogonal and QTA = R, where A is an M x N
    matrix, M >= N, and R is upper triangular. The upper triangular part
    of A is overwritten by R and the lower triangular part below the
    main diagonal is overwritten by the Householder vectors associated
    with the Householder matrices (Golub and Van Loan, 2013, Algorithm
    5.2.1, p. 249).

    Parameters
    ----------
    A : array 2D
        M x N matrix (M >= N) to be factored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[0] >= A.shape[1], 'A must be M x N, where M >= N'

    M = A.shape[0]
    N = A.shape[1]
    for j in range(N):
        v, beta = House_vector(x=A[j:,j], check_input=True)
        A[j:,j:] = House_matvec(A=A[j:,j:], v=v, beta=beta,
                               order='PA', check_input=True)
        if j < M:
            A[j+1:,j] = v[1:M-j+1]


def Q_from_A(A, check_input=True):
    '''
    Retrieve matrix Q from the lower triangle of the M x N
    matrix A computed with function House_QR.

    Parameters
    ----------
    A : array 2D
        Matrix returned by function House_QR.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Q : array 2D
        Orthogonal matrix formed by the product of N Householder
        matrices, where N is the number of columns of A.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'

    M = A.shape[0]
    N = A.shape[1]
    Q = np.eye(N=M, M=N)
    #Q = np.identity(M)
    #Q[:,N:] = 0
    for j in range(N-1,-1,-1):
        v = np.hstack([1, A[j+1:,j]])
        beta = 2/(1 + np.dot(A[j+1:,j],A[j+1:,j]))
        Q[j:,j:] = House_matvec(A=Q[j:,j:], v=v, beta=beta,
                                order='PA', check_input=True)

    return Q


# steepst decent method

def sd_lsearch(A, dobs, p0, tol, itmax):
    '''
    Solve a positive-definite linear system by using the
    method of steepest decent with exact line seach (Golub
    and Van Loan, 2013, Algorithm 11.3.1, p. 627)
    .

    Parameters:
    -----------
    A : array 2D
        Symmetric positive definite N x N matrix.
    dobs : array 1D
        Observed data vector with N elements.
    p0 : array 1D
        Initial approximation of the solution p.
    tol : float
        Positive scalar controlling the termination criterion.

    Returns:
    --------
    p : array 1D
        Solution of the linear system.
    dpred : array 1D
        Predicted data vector produced by p.
    residuals_L2_norm_values : list
        L2 norm of the residuals along the iterations.
    '''

    A = np.asarray(A)
    dobs = np.asarray(dobs)
    p0 = np.asarray(p0)
    assert A.shape[0] == A.shape[1], 'A must be square'
    assert dobs.size == A.shape[0] == p0.size, 'A order, dobs size and p0 size must be the same'
    assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'
    assert isinstance(itmax, int) & (itmax > 0), 'itmax must be a positive integer'

    N = dobs.size
    p = p0.copy()
    dpred = np.dot(A, p)
    # gradient is the residuals vector
    grad = dpred - dobs
    # Euclidean norm of the residuals
    residuals_norm_values = []
    for iteration in range(itmax):
        mu = np.dot(grad,grad)/np.linalg.multi_dot([grad, A, grad])
        p -= mu*grad
        dpred = np.dot(A, p)
        grad = dpred - dobs
        residuals_norm = np.linalg.norm(grad)
        residuals_norm_values.append(residuals_norm)
        if residuals_norm < tol:
            break
    return p, dpred, residuals_norm_values


# conjugate gradient method

def cg_method(A, dobs, p0, tol):
    '''
    Solve a positive-definite linear system by using the
    conjugate gradient method (Golub and Van Loan, 2013,
    modified Algorithm 11.3.3, p. 635).

    Parameters:
    -----------
    A : array 2D
        Symmetric positive definite N x N matrix.
    dobs : array 1D
        Observed data vector with N elements.
    p0 : array 1D
        Initial approximation of the solution p.
    tol : float
        Positive scalar controlling the termination criterion.

    Returns:
    --------
    p : array 1D
        Solution of the linear system.
    dpred : array 1D
        Predicted data vector produced by p.
    residuals_L2_norm_values : list
        L2 norm of the residuals along the iterations.
    '''

    A = np.asarray(A)
    dobs = np.asarray(dobs)
    p0 = np.asarray(p0)
    assert A.shape[0] == A.shape[1], 'A must be square'
    assert dobs.size == A.shape[0] == p0.size, 'A order, dobs size and p0 size must be the same'
    assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'

    N = dobs.size
    p = p0.copy()
    # residuals vector
    res = dobs - np.dot(A, p)
    # residuals L2 norm
    res_L2 = np.dot(res,res)
    # Euclidean norm of the residuals
    res_norm = np.sqrt(res_L2)
    # List of Euclidean norm of the residuals
    residuals_norm_values = [res_norm]
    # positive scalar controlling convergence
    delta = tol*np.linalg.norm(dobs)

    # iteration 1
    if res_norm > delta:
        q = res
        w = np.dot(A, q)
        mu = res_L2/np.dot(q,w)
        p += mu*q
        res -= mu*w
        res_L2_ = res_L2
        res_L2 = np.dot(res,res)
        res_norm = np.sqrt(res_L2)

    residuals_norm_values.append(res_norm)

    # remaining iterations
    while res_norm > delta:
        tau = res_L2/res_L2_
        q = res + tau*q
        w = np.dot(A, q)
        mu = res_L2/np.dot(q,w)
        p += mu*q
        res -= mu*w
        res_L2_ = res_L2
        res_L2 = np.dot(res,res)
        res_norm = np.sqrt(res_L2)
        residuals_norm_values.append(res_norm)

    dpred = np.dot(A,p)

    return p, dpred, residuals_norm_values


def cgnr_method(A, dobs, p0, tol):
    '''
    Solve a linear system by using the conjugate gradient
    normal equation residual method (Golub and Van Loan, 2013,
    modified Algorithm 11.3.3 according to Figure 11.3.1 ,
    p. 637).

    Parameters:
    -----------
    A : array 2D
        Rectangular N x M matrix.
    dobs : array 1D
        Observed data vector with N elements.
    p0 : array 1D
        Initial approximation of the M x 1 solution p.
    tol : float
        Positive scalar controlling the termination criterion.

    Returns:
    --------
    p : array 1D
        Solution of the linear system.
    dpred : array 1D
        Predicted data vector produced by p.
    residuals_L2_norm_values : list
        L2 norm of the residuals along the iterations.
    '''

    A = np.asarray(A)
    dobs = np.asarray(dobs)
    p0 = np.asarray(p0)
    assert dobs.size == A.shape[0], 'A order and dobs size must be the same'
    assert p0.size == A.shape[1], 'A order and p0 size must be the same'
    assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'

    N = dobs.size
    p = p0.copy()
    # residuals vector
    res = dobs - np.dot(A, p)

    # auxiliary variable
    z = np.dot(A.T, res)

    # L2 norm of z
    z_L2 = np.dot(z,z)
    # Euclidean norm of the residuals
    res_norm = np.linalg.norm(res)
    # List of Euclidean norm of the residuals
    residuals_norm_values = [res_norm]
    # positive scalar controlling convergence
    delta = tol*np.linalg.norm(dobs)

    # iteration 1
    if res_norm > delta:
        q = z
        w = np.dot(A, q)
        mu = z_L2/np.dot(w,w)
        p += mu*q
        res -= mu*w
        z = np.dot(A.T, res)
        z_L2_ = z_L2
        z_L2 = np.dot(z,z)
        res_norm = np.linalg.norm(res)

    residuals_norm_values.append(res_norm)

    # remaining iterations
    while res_norm > delta:
        tau = z_L2/z_L2_
        q = z + tau*q
        w = np.dot(A, q)
        mu = z_L2/np.dot(w,w)
        p += mu*q
        res -= mu*w
        z = np.dot(A.T, res)
        z_L2_ = z_L2
        z_L2 = np.dot(z,z)
        res_norm = np.linalg.norm(res)
        residuals_norm_values.append(res_norm)

    dpred = np.dot(A,p)

    return p, dpred, residuals_norm_values
