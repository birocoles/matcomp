import numpy as np
import scipy as sp
from scipy.linalg import dft
from scipy.fft import fft as spfft
from numpy.testing import assert_almost_equal as aae
import pytest
from .. import algorithms as mca

# parameter 'decimal' in assert_almost_equal
tol=10

# Fourier matrix

def test_DFT_matrix_compare_scipy():
    'compare DFT matrix with scipy.linalg.dft'
    N = 20
    # Fourier matrix of DFT
    reference_unscaled = dft(N, scale=None)
    reference_n = dft(N, scale='n')
    reference_sqrtn = dft(N, scale='sqrtn')
    computed_unscaled = mca.DFT_matrix(N, scale=None, conjugate=False)
    computed_n = mca.DFT_matrix(N, scale='n', conjugate=False)
    computed_sqrtn = mca.DFT_matrix(N, scale='sqrtn', conjugate=False)
    aae(computed_unscaled, reference_unscaled, decimal=tol)
    aae(computed_n, reference_n, decimal=tol)
    aae(computed_sqrtn, reference_sqrtn, decimal=tol)
    # Fourier matrix of IDFT
    computed_unscaled = mca.DFT_matrix(N, scale=None, conjugate=True)
    computed_n = mca.DFT_matrix(N, scale='n', conjugate=True)
    computed_sqrtn = mca.DFT_matrix(N, scale='sqrtn', conjugate=True)
    aae(computed_unscaled, np.conj(reference_unscaled), decimal=tol)
    aae(computed_n, np.conj(reference_n), decimal=tol)
    aae(computed_sqrtn, np.conj(reference_sqrtn), decimal=tol)


def test_DFT_matrix_invalid_scale():
    'check error for invalid scaling factor'
    data = np.zeros(10)
    invalid_scales = [data.size, np.sqrt(data.size), 'N', 'srqtn']
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            mca.DFT_matrix(N=data.size, scale=invalid_scale)


def test_DFT_matrix_even_odd_decomposition():
    'verify the decomposition into odd and even columns'
    # This decomposition is shown in Golub and Van Loan (2013, p. 35)

    # define an even number of data
    N = 20

    # Full DFT matrix
    FN = mca.DFT_matrix(N=N, scale=None)

    # Decomposed matrix
    Omega = np.exp(-1j*2*np.pi/N)
    Omega = np.diag(np.power(Omega, np.arange(N//2)))
    FN_half = mca.DFT_matrix(N=N//2, scale=None)
    FN_even = np.vstack([FN_half,
                         FN_half])
    FN_odd = np.vstack([np.dot( Omega, FN_half),
                        np.dot(-Omega, FN_half)])
    # compare the even columns
    aae(FN[:,0::2], FN_even, decimal=tol)
    # comapre the odd columns
    aae(FN[:,1::2], FN_odd, decimal=tol)

    # Test for conjugate matrix
    FN_even = np.vstack([np.conj(FN_half),
                         np.conj(FN_half)])
    FN_odd = np.vstack([np.dot( np.conj(Omega), np.conj(FN_half)),
                        np.dot(-np.conj(Omega), np.conj(FN_half))])
    # compare the even columns
    aae(np.conj(FN[:,0::2]), FN_even, decimal=tol)
    # comapre the odd columns
    aae(np.conj(FN[:,1::2]), FN_odd, decimal=tol)


# Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT)

def test_DFT_IDFT_invalid_scale():
    'check error for invalid scaling factor'
    data = np.zeros(10)
    invalid_scales = [data.size, np.sqrt(data.size), 'N', 'srqtn']
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            mca.DFT_dumb(x=data, scale=invalid_scale)
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            mca.IDFT_dumb(X=data, scale=invalid_scale)
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            mca.DFT_recursive(x=data, scale=invalid_scale)


def test_DFT_dumb_compare_scipy_fft_fft():
    'compare with scipy.fft.fft'
    np.random.seed(56)
    # scale=None
    data = np.random.rand(15)
    reference_output_scipy = spfft(x=data, norm=None)
    computed_output_dumb = mca.DFT_dumb(x=data, scale=None)
    aae(reference_output_scipy, computed_output_dumb, decimal=tol)
    # scale='sqrtn'
    data = np.random.rand(15)
    reference_output_scipy = spfft(x=data, norm='ortho')
    computed_output_dumb = mca.DFT_dumb(x=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_dumb, decimal=tol)


def test_DFT_recursive_compare_scipy_fft_fft():
    'compare with scipy.fft.fft'
    np.random.seed(56)
    # scale=None
    data = np.random.rand(2**7)
    reference_output_scipy = spfft(x=data, norm=None)
    computed_output_recursive = mca.DFT_recursive(x=data, scale=None)
    aae(reference_output_scipy, computed_output_recursive, decimal=tol)
    scale='sqrtn'
    data = np.random.rand(2**7)
    reference_output_scipy = spfft(x=data, norm='ortho')
    computed_output_recursive = mca.DFT_recursive(x=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_recursive, decimal=tol)


def test_IDFT_dumb_compare_scipy_fft_ifft():
    'compare with scipy.fft.ifft'
    np.random.seed(16)
    # scale=None
    data = np.random.rand(15)+1j*np.random.rand(15)
    reference_output_scipy = sp.fft.ifft(x=data, norm=None)
    computed_output_dumb = mca.IDFT_dumb(X=data, scale='n')
    aae(reference_output_scipy, computed_output_dumb, decimal=tol)
    scale='sqrtn'
    data = np.random.rand(15)+1j*np.random.rand(15)
    reference_output_scipy = sp.fft.ifft(x=data, norm='ortho')
    computed_output_dumb = mca.IDFT_dumb(X=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_dumb, decimal=tol)


def test_IDFT_recursive_compare_scipy_fft_ifft():
    'compare with scipy.fft.ifft'
    np.random.seed(70)
    # scale=None
    data = np.random.rand(2**7)+1j*np.random.rand(2**7)
    reference_output_scipy = sp.fft.ifft(x=data, norm=None)
    computed_output_recursive = mca.IDFT_recursive(X=data, scale='n')
    aae(reference_output_scipy, computed_output_recursive, decimal=tol)
    scale='sqrtn'
    data = np.random.rand(2**7)+1j*np.random.rand(2**7)
    reference_output_scipy = sp.fft.ifft(x=data, norm='ortho')
    computed_output_recursive = mca.IDFT_recursive(X=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_recursive, decimal=tol)


def test_DFT_parseval_theorem():
    'energy is the same in time/space and Fourier domains'
    np.random.seed(111)
    data = np.random.rand(2**4)
    X_dumb = mca.DFT_dumb(x=data, scale='sqrtn')
    X_recursive = mca.DFT_recursive(x=data, scale='sqrtn')
    energy_data = np.dot(a=data, b=data)
    energy_X_dumb = np.dot(a=X_dumb, b=np.conj(X_dumb)).real
    energy_X_recursive = np.dot(a=X_recursive, b=np.conj(X_recursive)).real
    aae(energy_data, energy_X_dumb, decimal=tol)
    aae(energy_data, energy_X_recursive, decimal=tol)

# Householder transformation

def test_House_vector_beta_specific_input():
    'verify beta value for speficif x[0] and sigma'

    # sigma = np.dot(x[1:], x[1:])

    # verify if for x.size = 1, v and beta are zero
    x = 6.2
    v, beta = mca.House_vector(x=x)
    aae(beta, 0, decimal=tol)

    # verify if for (sigma == 0) and (x[0] >= 0), beta = 0
    x = np.zeros(7)
    x[0] = 3
    v, beta = mca.House_vector(x=x)
    aae(beta, 0, decimal=tol)
    # verify if for (sigma == 0) and (x[0] < 0), beta = -2
    x = np.zeros(7)
    x[0] = -4
    v, beta = mca.House_vector(x=x)
    aae(beta, -2, decimal=tol)


def test_House_vector_parameter_beta():
    'verify that beta = 2/dot(v,v)'
    np.random.seed(23)
    a = np.random.rand(7)
    v, beta = mca.House_vector(x=a)
    aae(beta, 2/np.dot(v,v), decimal=tol)


def test_House_vector_orthogonal_reflection():
    'verify that the resulting Householder reflection is orthogonal'
    np.random.seed(14)
    # real vector
    a = np.random.rand(7)
    v, beta = mca.House_vector(x=a)
    P = np.identity(a.size) - beta*np.outer(v,v)
    aae(np.dot(P.T,P), np.dot(P,P.T), decimal=tol)
    aae(np.dot(P.T,P), np.identity(a.size), decimal=tol)


def test_House_vector_reflection_property():
    'verify that Px = norm_2(x) u_0'
    np.random.seed(43)
    # real vector
    x = np.random.rand(7)
    v, beta = mca.House_vector(x=x)
    # compute the Householder reflection
    P = np.identity(x.size) - beta*np.outer(v,v)

    x_norm_2 = np.linalg.norm(x)
    u_0 = np.zeros_like(x)
    u_0[0] = 1

    aae(np.dot(P,x), x_norm_2*u_0, decimal=tol)


def test_House_vector_matvec_matmat_reflection():
    'verify matrix-matrix product with Householder reflections'
    np.random.seed(32)
    # real vector
    N = 7
    a = np.random.rand(N)
    v, beta = mca.House_vector(x=a)
    P = np.identity(a.size) - beta*np.outer(v,v)
    A = np.random.rand(N,N)
    PA1 = np.dot(P, A)
    PA2 = mca.House_matvec(A=A, v=v, beta=beta, order='PA')
    aae(PA1, PA2, decimal=tol)
    AP1 = np.dot(A, P)
    AP2 = mca.House_matvec(A=A, v=v, beta=beta, order='AP')
    aae(AP1, AP2, decimal=tol)


# Givens transformation

def test_Givens_rotation_definition():
    'verify if Givens rotation satisfies its definition'
    np.random.rand(3)
    # general a and b
    a = 10*np.random.rand()
    b = 10*np.random.rand()
    c, s = mca.Givens_rotation(a=a, b=b)
    G = np.array([[ c, s],
                  [-s, c]])
    v = np.array([a, b])
    Gv = np.dot(G.T, v)
    aae(Gv[1], 0, decimal=tol)
    # b = 0
    a = 10*np.random.rand()
    b = 0
    c, s = mca.Givens_rotation(a=a, b=b)
    G = np.array([[ c, s],
                  [-s, c]])
    v = np.array([a, b])
    Gv = np.dot(G.T, v)
    aae(Gv[1], 0, decimal=tol)
    # |b| > |a|
    a = 7*np.random.rand()
    b = 10*np.random.rand()
    c, s = mca.Givens_rotation(a=a, b=b)
    G = np.array([[ c, s],
                  [-s, c]])
    v = np.array([a, b])
    Gv = np.dot(G.T, v)
    aae(Gv[1], 0, decimal=tol)


def test_Givens_matvec_matmat():
    'verify matrix-matrix product with Givens rotations'
    np.random.seed(3)
    M = 5
    N = 7
    A = np.round(np.random.rand(M,N), decimals=3)
    i = 2
    k = 3
    c, s = mca.Givens_rotation(a=A[i,3], b=A[k,3])
    # verify product GTA
    G = np.identity(M)
    G[i,i] = c
    G[i,k] = s
    G[k,i] = -s
    G[k,k] = c
    A2 = A.copy()
    mca.Givens_matvec(A=A2, c=c, s=s, i=i, k=k, order='GTA')
    aae(A2, np.dot(G.T,A), decimal=tol)
    # verify AG
    G = np.identity(N)
    G[i,i] = c
    G[i,k] = s
    G[k,i] = -s
    G[k,k] = c
    A2 = A.copy()
    mca.Givens_matvec(A=A2, c=c, s=s, i=i, k=k, order='AG')
    aae(A2, np.dot(A,G), decimal=tol)


def test_Givens_cs2rho_Givens_rho2cs():
    'verify consistency'
    np.random.seed(11)
    a = 10*np.random.rand()
    b = 10*np.random.rand()
    c, s = mca.Givens_rotation(a=a, b=b)
    rho = mca.Givens_cs2rho(c=c, s=s)
    c2, s2 = mca.Givens_rho2cs(rho=rho)
    aae(c, c2, decimal=tol)
    aae(s, s2, decimal=tol)


# QR decomposition

def test_QR_House_Q_from_QR_House_decomposition():
    'verify the computed Q and R matrices'
    np.random.seed(18)
    M = 7
    N = 7
    A = np.random.rand(M,N)
    A2 = A.copy()
    mca.QR_House(A2)
    Q = mca.Q_from_QR_House(A=A2)
    R = np.triu(A2)
    aae(A, Q@R, decimal=tol)


def test_QR_House_Q_from_QR_House_orthogonal():
    'verify the orthogonality of the computed Q'
    np.random.seed(18)
    M = 7
    N = 7
    A = np.random.rand(M,N)
    mca.QR_House(A)
    Q = mca.Q_from_QR_House(A=A)
    aae(np.identity(M), Q.T@Q, decimal=tol)


def test_QR_House_Cholesky():
    'verify that R is the transpose of the Cholesky factor of ATA'
    np.random.seed(18)
    M = 7
    N = 7
    A = np.random.rand(M,N)
    ATA = A.T@A
    mca.QR_House(A)
    R = np.triu(A)
    aae(ATA, R.T@R, decimal=tol)


def test_QR_Givens_Q_from_QR_Givens_decomposition():
    'verify the computed Q and R matrices'
    np.random.seed(18)
    M = 7
    N = 7
    A = np.random.rand(M,N)
    A2 = A.copy()
    mca.QR_Givens(A2)
    Q = mca.Q_from_QR_Givens(A=A2)
    R = np.triu(A2)
    aae(A, np.dot(Q, R), decimal=tol)


def test_QR_Givens_Q_from_QR_Givens_orthogonal():
    'verify the orthogonality of the computed Q'
    np.random.seed(18)
    M = 7
    N = 7
    A = np.random.rand(M,N)
    mca.QR_Givens(A)
    Q = mca.Q_from_QR_Givens(A=A)
    aae(np.identity(M), np.dot(Q.T,Q), decimal=tol)


def test_QR_Givens_Cholesky():
    'verify that R is the transpose of the Cholesky factor of ATA'
    np.random.seed(18)
    M = 7
    N = 7
    A = np.random.rand(M,N)
    ATA = np.dot(A.T,A)
    mca.QR_Givens(A)
    R = np.triu(A)
    aae(ATA, np.dot(R.T,R), decimal=tol)


def test_QR_MGS_decomposition():
    'verify the computed Q and R matrices'
    np.random.seed(6)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    Q1, R1 = mca.QR_MGS(A)
    aae(A, np.dot(Q1, R1), decimal=tol)


def test_QR_MGS_Q_orthogonal():
    'verify the orthogonality of the computed Q'
    np.random.seed(1)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    Q1, R1 = mca.QR_MGS(A)
    aae(np.identity(N), np.dot(Q1.T,Q1), decimal=tol)


def test_QR_MGS_Cholesky():
    'verify that R is the transpose of the Cholesky factor of ATA'
    np.random.seed(98)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    ATA = np.dot(A.T,A)
    Q1, R1 = mca.QR_MGS(A)
    aae(ATA, R1.T@R1, decimal=tol)


# Hessenberg QR step
def test_H_plus_RQ_decomposition():
    'verify if H_plus is actually RQ'
    # generate a square matrix
    np.random.seed(18)
    N = 7
    A = np.random.rand(N,N)
    # compute an upper Hessenberg matrix from A
    H = A.copy()
    mca.upper_Hessen_House(H)
    H = np.triu(H, k=-1)
    # compute the QR decomposition of H
    R = H.copy()
    mca.QR_Givens(R)
    Q = mca.Q_from_QR_Givens(R)
    R = np.triu(R)
    # compute H_plus
    H_plus = H.copy()
    mca.H_plus_RQ(H_plus)
    aae(H_plus, R@Q, decimal=tol)


def test_upper_Hessen_House_decomposition():
    'verify if the factored form retrives the original matrix'
    np.random.seed(567)
    N = 7
    A = np.random.rand(N,N)
    H = A.copy()
    mca.upper_Hessen_House(H)
    U0 = mca.U0_from_upper_Hessen_House(H)
    H = np.triu(H, k=-1)
    aae(A, np.linalg.multi_dot([U0, H, U0.T]), decimal=tol)


# Tridiagonalization
def test_House_tridiag_symmetry():
    'verify if the factored form is symmetric'
    np.random.seed(79)
    N = 7
    A = np.random.rand(N,N)
    A = A.T + A
    mca.House_tridiag(A)
    superdiag = np.diag(v=A,k=1)
    subdiag = np.diag(v=A,k=-1)
    aae(superdiag, subdiag, decimal=tol)


def test_House_tridiag_zeros():
    'verify if the upper elements are zero'
    np.random.seed(3)
    N = 7
    A = np.random.rand(N,N)
    A = A.T + A
    mca.House_tridiag(A)
    U = np.triu(A, k=2)
    aae(np.zeros((N,N)), U, decimal=tol)


def test_House_tridiag_decomposition():
    'verify if the factored form retrives the original matrix'
    np.random.seed(5)
    N = 7
    A = np.random.rand(N,N)
    A = A.T + A
    Tridiag = A.copy()
    mca.House_tridiag(Tridiag)
    Q = mca.Q_from_House_tridiag(Tridiag)
    Tridiag = np.triu(Tridiag, k=-1)
    aae(A, np.linalg.multi_dot([Q, Tridiag, Q.T]), decimal=tol)


# Bidiagonalization
def test_Golub_Kahan_bidiag_decomposition():
    'verify if the factored form retrieves the original matrix'
    np.random.seed(72)
    M = 7
    N = 4
    A = np.random.rand(M,N)
    alpha, beta, U, V = mca.Golub_Kahan_bidiag(A)
    # create the bidiagonal matrix
    # without its last M-N zero rows
    B = np.diag(v=alpha, k=0) + np.diag(v=beta, k=1)
    B2 = np.dot(U.T,A).dot(V)
    aae(B, B2, decimal=tol)


def test_Golub_Kahan_bidiag_U_orthogonal():
    'verify if the computed U are orthogonal'
    np.random.seed(42)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    alpha, beta, U, V = mca.Golub_Kahan_bidiag(A)
    aae(np.identity(N), np.dot(U.T,U), decimal=tol)


def test_Golub_Kahan_bidiag_V_orthogonal():
    'verify if the computed V are orthogonal'
    np.random.seed(42)
    M = 7
    N = 5
    A = np.random.rand(M,N)
    alpha, beta, U, V = mca.Golub_Kahan_bidiag(A)
    aae(np.identity(N), np.dot(V.T,V), decimal=tol)
