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
    Compute the real N x 1 Householder vector (Golub and Van Loan,
    2013, Algorithm 5.1.1, p. 236) v, with v[0] = 1, such that N x N
    matrix P = I - beta outer(v,v) (Householder reflection) is
    orthogonal and Px = norm_2(x) u_0, where u_0 is an N x 1 vector
    with all elements equal to zero, except the 0th, that is equal
    to 1.

    Parameters
    ----------
    x : array 1D
        N x 1 vector perpendicular to the Householder vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    v : array 1D
        Householder vector.

    beta : float
        Scalar equal to 1/dot(v,v).
    '''
    x = np.atleast_1d(x)
    if check_input is True:
        assert x.ndim == 1, 'x must be a vector'
        #assert x.size > 1, 'x size must be greater than 1'

    N = x.size

    if N > 1:
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
            beta = 2*(v[0]**2)/(sigma + v[0]**2)
            v /= v[0]
    else:
        v = np.array([0.])
        beta = np.array([0.])

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
    Update matrix A with the matrix-matrix product AG or GTA, where
    G is a Givens rotation G(i, k, theta) (Golub and Van Loan, 2013,
    p. 241).

    Parameters
    ----------
    A : array 2D
        Matrix to be updated.

    c, s : scalars
        Cosine and Sine of theta forming the Givens rotation matrix.

    i, k : ints
        Indices of the Givens rotation matrix.

    order : string
        If 'AG', it defines the product AG. If 'GTA',
        it defines the product GTA. Default is 'AG'.

    check_input : boolean
        If True, verify if the input is valid. Default is True.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert np.isscalar(c) and np.isscalar(s), 'c and s must be scalars'
        assert isinstance(i, int) and (i >= 0), 'i must be a an integer >= 0'
        assert isinstance(k, int) and (k >= 0), 'k must be a an integer >= 0'
        assert np.allclose((c**2 + s**2), 1), 'c**2 + s**2 must be equal to 1'
        assert order in ['AG', 'GTA'], "order must be 'AG' or 'GTA'"

    M = A.shape[0]
    N = A.shape[1]
    G = np.array([[ c, s],
                  [-s, c]])
    if order is 'AG':
        assert (i <= N) and (k <= N), 'i and k must be < N'
        A[:,[i,k]] = np.dot(A[:,[i,k]], G)
    else:
        assert (i <= M) and (k <= M), 'i and k must be < M'
        A[[i,k],:] = np.dot(G.T, A[[i,k],:])


def Givens_cs2rho(c, s, check_input=True):
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
        assert np.allclose((c**2 + s**2), 1), 'c**2 + s**2 must be equal to 1'

    if c == 0:
        rho = 1
    elif np.abs(s) < np.abs(c):
        rho = np.sign(c)*s/2
    else:
        rho = 2*np.sign(s)/c

    return rho


def Givens_rho2cs(rho, check_input=True):
    '''
    Compute c and s from the representing scalar rho
    obtained with function Givens_cs2rho (Golub and Van Loan, 2013, p. 242).

    Parameters
    ----------
    rho : scalar
        Representing scalar obtained with function Givens_cs2rho.

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
def QR_House(A, check_input=True):
    '''
    Compute the M x M Householder matrices Hj such that
    Q = H0 H1 ... HN-1 is an M x M orthogonal matrix and QTA = R,
    where A is an M x N matrix, M >= N, and R is an M x N upper
    triangular matrix. The upper triangular part
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
        v, beta = House_vector(x=A[j:,j], check_input=False)
        A[j:,j:] = House_matvec(A=A[j:,j:], v=v, beta=beta,
                                order='PA', check_input=False)
        # if j < M:
        #     A[j+1:,j] = v[1:M-j+2]
        # j is always lower than M
        # A[j+1:,j] = v[1:M-j+2]
        A[j+1:,j] = v[1:]


def Q_from_QR_House(A, check_input=True):
    '''
    Retrieve the M x M matrix Q from the lower triangle of the
    M x N matrix A computed with function QR_House.

    Parameters
    ----------
    A : array 2D
        M x N matrix returned by function QR_House.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Q : array 2D
        M x M orthogonal matrix formed by the product of
        N M x M Householder matrices, where N is the number
        of columns of A.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'

    M = A.shape[0]
    N = A.shape[1]
    # Compute the full M x M Q matrix
    Q = np.identity(M)
    for j in range(N-2,-1,-1):
        v = np.hstack([1, A[j+1:,j]])
        beta = 2/(1 + np.dot(A[j+1:,j],A[j+1:,j]))
        Q[j:,j:] = House_matvec(A=Q[j:,j:], v=v, beta=beta,
                                order='PA', check_input=False)

    return Q


def QR_Givens(A, check_input=True):
    '''
    Compute the Givens rotations Gj such that
    Q = G0G1...Gt is an M x M orthogonal matrix and QTA = R,
    where A is an M x N matrix, M >= N, and R is an M x N upper
    triangular matrix. The upper triangular part
    of A is overwritten by R and the lower triangular part below the
    main diagonal is overwritten by the representing scalar obtained with
    function cs2rho (Golub and Van Loan, 2013, Algorithm 5.2.4, p. 252).

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
        for i in range(M-1, j, -1):
            c, s = Givens_rotation(a=A[i-1,j], b=A[i,j], check_input=False)
            #Givens_matvec(A, c, s, i-1, i, order='GTA', check_input=False)
            # By passing only the j: columns, the stored representing factors
            # aren't 'destroyed' by the algorithm
            Givens_matvec(A[:,j:], c, s, i-1, i, order='GTA', check_input=False)
            A[i,j] = Givens_cs2rho(c, s, check_input=False)


def Q_from_QR_Givens(A, check_input=True):
    '''
    Retrieve the M x M matrix Q from the lower triangle of the
    M x N matrix A computed with function QR_Givens by using the
    function rho2cs.

    Parameters
    ----------
    A : array 2D
        M x N matrix returned by function QR_Givens.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Q : array 2D
        M x M orthogonal matrix formed by the product of N
        M x M Givens rotation matrices, where N is the number
        of columns of A.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'

    M = A.shape[0]
    N = A.shape[1]
    Q = np.identity(M)
    for j in range(N):
        for i in range(M-1, j, -1):
            c, s = Givens_rho2cs(rho=A[i,j], check_input=False)
            Givens_matvec(Q, c, s, i-1, i, order='AG', check_input=False)

    return Q


def QR_MGS(A, check_input=True):
    '''
    Compute the QR factorization of a full-column rank M x N
    matrix A, where M >= N, Q is an M x N orthogonal matrix and R is
    an N x N upper triangular matrix by applying the Modified
    Gram-Schmidt method (Golub and Van Loan, 2013, Algorithm 5.2.6,
    p. 255).

    Parameters
    ----------
    A : array 2D
        Full-column rank M x N matrix (M >= N) to be factored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Q : array 2D
         M x N orthogonal matrix.

    R : array 2D
        N x N upper triangular matrix.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[0] >= A.shape[1], 'A must be M x N, where M >= N'

    M = A.shape[0]
    N = A.shape[1]

    Q = A.copy()
    R = np.zeros((N,N))
    for k in range(N):
        R[k,k] = np.linalg.norm(Q[:,k])
        Q[:,k] = Q[:,k]/R[k,k]
        for j in range(k+1,N):
            R[k,j] = np.dot(Q[:,k],Q[:,j])
            Q[:,j] -= R[k,j]*Q[:,k]

    return Q, R


# Hessenberg QR step
def H_plus_RQ(H, check_input=True):
    '''
    If H is an N x N upper Hessenberg matrix, then this algorithm overwrites
    H with H+ = RQ where H = QR is the QR factorization of H (Golub and Van
    Loan, 2013, Algorithm 7.4.1, p. 378).

    Parameters
    ----------
    H : array 2D
        N x N upper Hessenberg matrix to be factored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.
    '''
    H = np.asarray(H)
    if check_input is True:
        assert H.ndim == 2, 'H must be a matrix'
        assert H.shape[0] == H.shape[1], 'H must be square'

    N = H.shape[0]

    for k in range(N-1):
        c, s = Givens_rotation(a=H[k,k], b=H[k+1,k], check_input=False)
        Givens_matvec(H, c, s, k, k+1, order='GTA', check_input=False)
        Givens_matvec(H, c, s, k, k+1, order='AG', check_input=False)
    # for k in range(N-1):
    #     Givens_matvec(H, c, s, k, k+1, order='AG', check_input=False)


def upper_Hessen_House(A, check_input=True):
    '''
    Given an N x N matrix A, overwrites A with an upper Hessenberg matrix
    H = U0^T A U0 , where U0 is a product of Householder matrices
    (Golub and Van Loan, 2013, Algorithm 7.4.2, p. 379). The Householder
    vectors are stored in the lower triangle of A.

    Parameters
    ----------
    A : array 2D
        N x N matrix to be factored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[0] == A.shape[1], 'A must be square'

    N = A.shape[0]

    for k in range(N-1):
        v, beta = House_vector(x=A[k+1:,k], check_input=False)
        A[k+1:,k:] = House_matvec(A=A[k+1:,k:], v=v, beta=beta,
                                  order='PA', check_input=False)
        A[:,k+1:] = House_matvec(A=A[:,k+1:], v=v, beta=beta,
                                 order='AP', check_input=False)
        # store the Householder vectors in the lower triangle of A
        A[k+2:,k] = v[1:]


def U0_from_upper_Hessen_House(A, check_input=True):
    '''
    Retrieve the N x N matrix U0 from the lower triangle of the
    N x N matrix A computed with function upper_Hessen_House.

    Parameters
    ----------
    A : array 2D
        N x N matrix returned by function upper_Hessen_House.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    U0 : array 2D
        N x N orthogonal matrix formed by the product of
        (N-2) N x N Householder matrices.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'

    N = A.shape[1]

    U0 = np.identity(N)
    for k in range(N-3,-1,-1):
    #for k in range(N-2):
        v = np.hstack([1, A[k+2:,k]])
        beta = 2/(1 + np.dot(A[k+2:,k],A[k+2:,k]))
        U0[k+1:,k:] = House_matvec(A=U0[k+1:,k:], v=v, beta=beta,
                                   order='PA', check_input=False)

    return U0


def Francis_QR_step(H, check_input=True, return_v=True):
    '''
    Given the an N x N real unreduced upper Hessenberg matrix H whose trailing
    2-by-2 principal submatrix has eigenvalues a1 and a2, this algorithm
    overwrites H with Z^T H Z, where Z is a product of Householder matrices
    and Z^T (H - a1 I)(H - a2 I) is upper triangular (Golub and Van Loan, 2013,
    Algorithm 7.5.1, p. 390).

    Parameters
    ----------
    H : array 2D
        N x N upper Hessenberg matrix to be factored. Possible complex
        components are ignored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    return_v : boolean
        If True, return a list containing the Householder vectors v without
        their first elements.
    '''
    H = np.asarray(H).real
    if check_input is True:
        assert H.ndim == 2, 'H must be a matrix'
        assert H.shape[0] == H.shape[1], 'H must be square'
        assert np.all(np.diag(v=H, k=-1) != 0), 'H must be unreduced'

    if return_v is True:
        list_v = []

    N = H.shape[0]

    # Compute first column of M = (H - a1 I)(H - a2 I)
    s = H[-2,-2] + H[-1,-1]
    t = H[-2,-2]*H[-1,-1] - H[-2,-1]*H[-1,-2]
    x = H[0,0]*H[0,0] + H[0,1]*H[1,0] - s*H[0,0] + t
    y = H[1,0]*(H[0,0] + H[1,1] - s)
    z = H[1,0]*H[2,1]

    # Compute the product P0 H P0
    v, beta = House_vector([x,y,z])
    H[:3,:] = House_matvec(A=H[:3,:], v=v, beta=beta, order='PA')
    H[:4,:3] = House_matvec(A=H[:4,:3], v=v, beta=beta, order='AP')

    if return_v is True:
        list_v.append(v[1:])

    # Compute the products from column k = 0 to N-4
    for k in range(N-3):

        x = H[k+1,k]
        y = H[k+2,k]
        z = H[k+3,k]

        v, beta = House_vector([x,y,z])

        H[k+1:k+4,k:] = House_matvec(A=H[k+1:k+4,k:],
                                     v=v, beta=beta, order='PA')
        H[:k+5,k+1:k+4] = House_matvec(H[:k+5,k+1:k+4],
                                       v=v, beta=beta, order='AP')

        if return_v is True:
            list_v.append(v[1:])

    # Compute the product for column k = N-3
    x = H[N-2, N-3]
    y = H[N-1, N-3]

    v, beta = House_vector([x,y])

    H[N-2:,N-3:] = House_matvec(A=H[N-2:,N-3:], v=v, beta=beta, order='PA')
    H[:,N-2:] = House_matvec(A=H[:,N-2:], v=v, beta=beta, order='AP')

    if return_v is True:
        list_v.append(v[1:])
        return list_v


def Z_from_Francis_QR_step(list_v, check_input=True):
    '''
    Retrieve the N x N matrix Z from the list of Householder vectors returned
    by the function Francis_QR_step.

    Parameters
    ----------
    list_v : list
        N-2 Householder vectors (without their first elements) computed by the
        function Francis_QR_step.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Z : array 2D
        N x N orthogonal matrix formed by the product of N-2 N x N Householder
        matrices.
    '''

    N_2 = len(list_v)
    if check_input is True:
        for vector in list_v[:N_2-1]:
            assert vector.size == 2, 'The first vector must have 2 elements'
        assert list_v[-1].size == 1, 'The last vector must have 1 element'

    # Compute the full N x N matrix Z
    Z = np.identity(N_2+2)
    for k, vector in enumerate(list_v(N_2-1)):
        v = np.hstack([1, vector])
        beta = 2/(1 + np.dot(vector,vector))
        Q[:,j:] = House_matvec(A=Q[j:,j:], v=v, beta=beta,
                                order='PA', check_input=False)

    return Q


# Bidiagonalization

# def House_bidiag(A, check_input=True):
#     '''
#     Given an M x N matrix A, the algorithm overwrites A with UTAV = B,
#     where B is upper bidiagonal, and U and V are orthogonal matrices
#     defined as products of Householder matrices (Golub and Van Loan, 2013,
#     Algorithm 5.4.2, p. 284-285).
#
#     Parameters
#     ----------
#     A : array 2D
#         Matrix used to compute the product.
#
#     check_input : boolean
#         If True, verify if the input is valid. Default is True.
#     '''
#     A = np.asarray(A)
#     if check_input is True:
#         assert A.ndim == 2, 'A must be a matrix'
#         assert A.shape[0] >= A.shape[1], 'A must be M x N, where M >= N'
#
#     M = A.shape[0]
#     N = A.shape[1]
#
#     for j in range(N):
#         v, beta = House_vector(A[j:,j])
#         A[j:,j:] = House_matvec(A=A[j:,j:], v=v, beta=beta,
#                                 order='PA', check_input=False)
#         A[j+1:,j] = v[1:M-j+2]
#         if j <= N-2:
#             v, beta = House_vector(A[j,j+1:])
#             A[j:,j+1:] = House_matvec(A=A[j:,j+1:], v=v, beta=beta,
#                                       order='AP', check_input=False)
#             A[j,j+2:] = v[1:N-j+1]
#
#
# def UV_from_B(B, check_input=True):
#     '''
#     Retrieve the M x N orthogonal matrix U and the N x N
#     orthogonal matrix V from the lower and upper triangle
#     of B computed with function House_bidiag.
#
#     Parameters
#     ----------
#     B : array 2D
#         M x N bidiagonal matrix computed with House_bidiag.
#
#     check_input : boolean
#         If True, verify if the input is valid. Default is True.
#
#     Returns
#     -------
#     U : array 2D
#         M x M orthogonal matrix formed by the product of
#         N M x M Householder matrices, where N is the number
#         of columns of B.
#
#     V : array 2D
#         N x N orthogonal matrix formed by the product of
#         N-2 N x N Householder matrices, where N is the number
#         of columns of B.
#     '''
#     B = np.asarray(B)
#     if check_input is True:
#         assert B.ndim == 2, 'B must be a matrix'
#
#     M = B.shape[0]
#     N = B.shape[1]
#     U = np.identity(M)
#     for j in range(N):
#         v = np.hstack([1, B[j+1:,j]])
#         beta = 2/(1 + np.dot(B[j+1:,j],B[j+1:,j]))
#         U[j:,j:] = House_matvec(A=U[j:,j:], v=v, beta=beta,
#                                 order='PA', check_input=False)
#
#     return U


def Golub_Kahan_bidiag(A, check_input=True):
    '''
    Given an M x N full-column rank matrix A, the algorithm
    computes the non-null elements of an upper bidiagonal matrix
    B = UBTAVB, where UB is an M x M orthogonal matrix and VB is an
    N x N orthogonal matrix (Golub and Van Loan, 2013, Algorithm
    10.4.1, p. 572). For convenience, the last column of UB is not
    computed, so that B[:-1] = UBTAVB.

    Parameters
    ----------
    A : array 2D
        M x N full-column rank matrix to be factored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    alpha : array 2D
        N x 1 vector containing the main diagonal of B.

    beta : array 2D
        (N-1) x 1 vector containing the upper diagonal of B.

    UB : array 2D
         M x N orthogonal matrix (the last columns are not computed).

    VB : array 2D
         N x N orthogonal matrix.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[0] >= A.shape[1], 'A must be M x N, where M >= N'

    M = A.shape[0]
    N = A.shape[1]

    k = 0
    beta = 1
    beta_list = []
    alpha_list = []
    v = np.zeros(N)
    v[0] = 1
    VB = []
    u = np.zeros(M)
    UB = []
    p = v
    for k in range(N):
        v = p/beta
        VB.append(v)
        r = np.dot(A,v) - beta*u
        alpha = np.linalg.norm(r)
        alpha_list.append(alpha)
        u = r/alpha
        UB.append(u)
        p = np.dot(A.T,u) - alpha*v
        beta = np.linalg.norm(p)
        beta_list.append(beta)

    alpha_list = np.array(alpha_list)
    beta_list = np.array(beta_list)[:-1]
    UB = np.vstack(UB).T
    VB = np.vstack(VB).T

    return alpha_list, beta_list, UB, VB


# Tridiagonalization

def House_tridiag(A, check_input=True):
    '''
    Given an N x N symmetric matrix A, the algorithm overwrites
    A with a tridiagonal matrix T = Q^T A Q, where Q is the product
    of Householder matrices H1H2...HN-2 (Golub and Van Loan, 2013,
    Algorithm 8.3.1, p. 459). The Householder vectors are stored in
    the lower triangle of A.

    Parameters
    ----------
    A : array 2D
        N x N symmetric matrix to be factored.

    check_input : boolean
        If True, verify if the input is valid. Default is True.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert np.all(A.T == A), 'A must be symmetric'

    N = A.shape[0]

    for k in range(N-2):
        v, beta = House_vector(A[k+1:,k])

        # original algoritmh from book
        # it doesn't work for me
        # p = beta*np.dot(A[k+1:,k+1:],v)
        # w = p + (beta*np.dot(p,v)/2)*v
        # A[k+1,k] = np.linalg.norm(A[k+1:,k])
        # A[k,k+1] = A[k+1,k]
        # A[k+1:,k+1:] = A[k+1:,k+1:] - np.outer(v,w) - np.outer(w,v)

        # zero kth column
        A[k+1:,k:] = House_matvec(A=A[k+1:,k:], v=v, beta=beta,
                                  order='PA', check_input= True)
        # zero kth row
        A[k:,k+1:] = House_matvec(A=A[k:,k+1:], v=v, beta=beta,
                                  order='AP', check_input=True)
        # store the Householder vectors in the lower triangle of A
        A[k+2:,k] = v[1:]


def Q_from_House_tridiag(A, check_input=True):
    '''
    Retrieve the N x N matrix Q from the lower triangle of the
    N x N matrix A computed with function House_tridiag.

    Parameters
    ----------
    A : array 2D
        N x N matrix returned by function House_tridiag.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Q : array 2D
        N x N orthogonal matrix formed by the product of
        (N-2) N x N Householder matrices.
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'

    N = A.shape[1]

    Q = np.identity(N)
    for k in range(N-3,-1,-1):
    #for k in range(N-2):
        v = np.hstack([1, A[k+2:,k]])
        beta = 2/(1 + np.dot(A[k+2:,k],A[k+2:,k]))
        Q[k+1:,k:] = House_matvec(A=Q[k+1:,k:], v=v, beta=beta,
                                  order='PA', check_input=False)

    return Q


# def Lanczos_tridiag():
#     '''
#     Given an N x N symmetric matrix A, the algorithm computes an
#     orthonormal matrix Q and and a symmetric tridiagonal matrix T
#     such that ... (Golub and Van Loan, 2013, Algorithm 10.1.1, p. 549).
#
#     Parameters
#     ----------
#     A : array 2D
#         N x N symmetric matrix to be factored.
#
#     check_input : boolean
#         If True, verify if the input is valid. Default is True.
#
#     '''
#     A = np.asarray(A)
#     if check_input is True:
#         assert A.ndim == 2, 'A must be a matrix'
#         assert np.all(A.T == A), 'A must be symmetric'
#
#     N = A.shape[0]
#
#     Q = np.identity(N)
#     for k in range(N-2):
#         v, beta = House_vector(A[k+1:,k])
#         Q[k+1:,k+1:] = House_matvec(A=Q[k+1:,k+1:], v=v, beta=beta,
#                                     order='AP', check_input=True)
#         p = beta*np.dot(A[k+1:,k+1:],v)
#         # w = p - (beta*np.dot(p,v)/2)*v
#         W = beta*np.dot(p,v)*np.outer(v,v)
#         A[k+1,k] = np.linalg.norm(A[k+1:,k])
#         A[k,k+1] = A[k+1,k]
#         # A[k+1:,k+1:] = A[k+1:,k+1:] - np.outer(v,w) - np.outer(w,v)
#         A[k+1:,k+1:] = A[k+1:,k+1:] + W - np.outer(v,p) - np.outer(p,v)
#         # store the Householder vectors
#         #A[k+2:,k] = v[1:]
#
#     return Q


# Symmetric QR step
def imp_symm_QR_step_shift(T, check_input=True):
    '''
    Given an unreduced symmetric tridiagonal N x N matrix T, the algorithm
    computes a matrix Z = G1 G2 ... GN-1 defined by the product of
    Givens rotations. The matrix Z^T(T - mu I) is upper triangular and mu is
    that eigenvalue of T's trailing 2-by-2 principal submatrix close to tnn.
    (Golub and Van Loan, 2013, Algorithm 8.3.2, p. 462).

    Parameters
    ----------
    T : array 2D
        N x N symmetric tridiagonal matrix to be factored.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    Z : array 2D
        N x N orthogonal matrix formed by the product of
        (N-1) N x N Givens rotations.
    mu : float
        Eigenvalue of T's trailing 2-by-2 principal submatrix close to tnn.
    '''
    T = np.asarray(T)
    if check_input is True:
        assert T.ndim == 2, 'T must be a matrix'
        assert np.all(np.diag(v=T,k=1) == np.diag(v=T,k=-1)), 'super and \
subdiagonals must be equal to each other'

    # compute the eigenvalue mu
    d = 0.5*(T[-2,-2] - T[-1,-1])
    mu = T[-1,-1] - (T[-1,-2]**2)/(d + np.sign(d)*np.sqrt(d**2 + T[-1,-2]**2))

    # set starting values for x and z
    x = T[0,0] - mu
    z = T[1,0]

    N = T.shape[0]
    Z = np.identity(N)
    for k in range(N-2):
        c, s = Givens_rotation(a=x, b=z, check_input=True)
        Givens_matvec(Z, c, s, k, k+1, order='AG', check_input=True)
        #Givens_matvec(Z, c, s, k, k+1, order='GTA', check_input=True)
        x = Z[k+1,k]
        z = Z[k+2,k]
    # k = N-2
    c, s = Givens_rotation(a=x, b=z, check_input=True)
    Givens_matvec(Z, c, s, k, k+1, order='AG', check_input=True)
    #Givens_matvec(Z, c, s, k, k+1, order='GTA', check_input=True)

    return Z, mu



# Singular Value Decomposition
def Golub_Kahan_SVD_step(alpha, beta, check_input=True):
    '''
    '''
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    if check_input is True:
        assert alpha.ndim == 1, 'alpha must be a vector'
        assert beta.ndim == 1, 'beta must be a vector'
        assert alpha.size == beta.size+1, 'alpha size must be = beta size + 1'
        assert np.all(alpha != 0), 'alpha cannot have zeros'
        assert np.all(beta != 0), 'beta cannot have zeros'

    B = np.diag(v=alpha, k=0) + np.diag(v=beta, k=1)
    T = np.dot(B.T,B)

    a_n_1 = T[-2,-2]
    b_n_1 = T[-2,-1]
    a_n   = T[-1,-1]
    d = (a_n_1 - a_n)/2
    mu = a_n + d - np.sign(d)*np.sqrt(d**2 + b_n_1**2)

    y = T[0,0] - mu
    z = T[0,1]

    N = alpha.size

    for k in range(N-2):
        c, s = Givens_rotation(a=y, b=z, check_input=True)
        Givens_matvec(B, c, s, k, k+1, order='AG', check_input=True)
        y = B[k,k]
        z = B[k+1,k]
        c, s = Givens_rotation(a=y, b=z, check_input=True)
        Givens_matvec(B, c, s, k, k+1, order='GTA', check_input=True)
        y = B[k,k+1]
        z = B[k,k+2]

    k = N-2
    c, s = Givens_rotation(a=y, b=z, check_input=True)
    Givens_matvec(B, c, s, k, k+1, order='AG', check_input=True)
    y = B[k,k]
    z = B[k+1,k]
    c, s = Givens_rotation(a=y, b=z, check_input=True)
    Givens_matvec(B, c, s, k, k+1, order='GTA', check_input=True)

    return B


def SVD(A, epsilon, check_input=True):
    '''
    '''
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[0] >= A.shape[1], 'A must be M x N, where M >= N'
        assert np.isscalar(epsilon), 'epsilon must be a scalar'
        assert (epsilon > 0) and (epsilon < 1), 'epsilon must be > 0 and < 1'

    M = A.shape[0]
    N = A.shape[1]

    # Compute the bidiagonal matrix B from A
    alpha_B, beta_B, U_B, V_B = Golub_Kahan_bidiag(A)

    # Create B
    B = np.diag(v=alpha_B,k=0) + np.diag(v=beta_B,k=1)

    q = 0
    while q != N:
        #print(q)
        # Set B[i, i+1] = 0
        for i in range(N-1):
            condition = epsilon*(np.abs(B[i,i]) + np.abs(B[i+1,i+1]))
            if np.abs(B[i, i+1]) <= condition:
                B[i, i+1] = 0
        # smallest p and largest q defining B11, B22, and B33
        aux = np.nonzero(np.diag(B,k=1))[0]
        if aux.size >= 2:
            p, q = aux[[0, -1]]
            q = N - q - 2
        else:
            return B

        if q < N:
            # indices of null elements in diagonal of B22
            zeros_diag_B22 = np.where(np.diag(B[p:N-q,p:N-q]) == 0)[0]
            if zeros_diag_B22.size > 0:
                # set corresponding superdiagonal elements equal to zero
                B[p:N-q,p:N-q][zeros_diag_B22,zeros_diag_B22+1] = 0
            else:
                alpha_B22 = np.diag(B[p:N-q,p:N-q], k=0)
                beta_B22 = np.diag(B[p:N-q,p:N-q], k=1)
                B[p:N-q,p:N-q] = Golub_Kahan_SVD_step(alpha=alpha_B22,
                                                      beta=beta_B22)
    return B


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
