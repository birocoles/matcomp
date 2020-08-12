import numpy as np
import scipy as sp
from scipy.linalg import dft
from numpy.testing import assert_almost_equal as aae
import pytest
import matcomp_algorithms as mca

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
    aae(computed_unscaled, reference_unscaled, decimal=10)
    aae(computed_n, reference_n, decimal=10)
    aae(computed_sqrtn, reference_sqrtn, decimal=10)
    # Fourier matrix of IDFT
    computed_unscaled = mca.DFT_matrix(N, scale=None, conjugate=True)
    computed_n = mca.DFT_matrix(N, scale='n', conjugate=True)
    computed_sqrtn = mca.DFT_matrix(N, scale='sqrtn', conjugate=True)
    aae(computed_unscaled, np.conj(reference_unscaled), decimal=10)
    aae(computed_n, np.conj(reference_n), decimal=10)
    aae(computed_sqrtn, np.conj(reference_sqrtn), decimal=10)


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
    aae(FN[:,0::2], FN_even, decimal=10)
    # comapre the odd columns
    aae(FN[:,1::2], FN_odd, decimal=10)

    # Test for conjugate matrix
    FN_even = np.vstack([np.conj(FN_half),
                         np.conj(FN_half)])
    FN_odd = np.vstack([np.dot( np.conj(Omega), np.conj(FN_half)),
                        np.dot(-np.conj(Omega), np.conj(FN_half))])
    # compare the even columns
    aae(np.conj(FN[:,0::2]), FN_even, decimal=10)
    # comapre the odd columns
    aae(np.conj(FN[:,1::2]), FN_odd, decimal=10)


# Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT)

def test_DFT_IDFT_invalid_scale():
    'check error for invalid scaling factor'
    data = np.zeros(10)
    invalid_scales = [data.size, np.sqrt(data.size), 'N', 'srqtn']
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            mca.DFT_dumb(x=data, scale=invalid_scale)
            mca.IDFT_dumb(X=data, scale=invalid_scale)
            mca.DFT_recursive(x=data, scale=invalid_scale)


def test_DFT_dumb_compare_scipy_fft_fft():
    'compare with scipy.fft.fft'
    np.random.seed(56)
    # scale=None
    data = np.random.rand(15)
    reference_output_scipy = sp.fft.fft(x=data, norm=None)
    computed_output_dumb = mca.DFT_dumb(x=data, scale=None)
    aae(reference_output_scipy, computed_output_dumb, decimal=10)
    # scale='sqrtn'
    data = np.random.rand(15)
    reference_output_scipy = sp.fft.fft(x=data, norm='ortho')
    computed_output_dumb = mca.DFT_dumb(x=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_dumb, decimal=10)


def test_DFT_recursive_compare_scipy_fft_fft():
    'compare with scipy.fft.fft'
    np.random.seed(56)
    # scale=None
    data = np.random.rand(2**7)
    reference_output_scipy = sp.fft.fft(x=data, norm=None)
    computed_output_recursive = mca.DFT_recursive(x=data, scale=None)
    aae(reference_output_scipy, computed_output_recursive, decimal=10)
    scale='sqrtn'
    data = np.random.rand(2**7)
    reference_output_scipy = sp.fft.fft(x=data, norm='ortho')
    computed_output_recursive = mca.DFT_recursive(x=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_recursive, decimal=10)


def test_IDFT_dumb_compare_scipy_fft_ifft():
    'compare with scipy.fft.ifft'
    np.random.seed(16)
    # scale=None
    data = np.random.rand(15)+1j*np.random.rand(15)
    reference_output_scipy = sp.fft.ifft(x=data, norm=None)
    computed_output_dumb = mca.IDFT_dumb(X=data, scale='n')
    aae(reference_output_scipy, computed_output_dumb, decimal=10)
    scale='sqrtn'
    data = np.random.rand(15)+1j*np.random.rand(15)
    reference_output_scipy = sp.fft.ifft(x=data, norm='ortho')
    computed_output_dumb = mca.IDFT_dumb(X=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_dumb, decimal=10)


def test_IDFT_recursive_compare_scipy_fft_ifft():
    'compare with scipy.fft.ifft'
    np.random.seed(70)
    # scale=None
    data = np.random.rand(2**7)+1j*np.random.rand(2**7)
    reference_output_scipy = sp.fft.ifft(x=data, norm=None)
    computed_output_recursive = mca.IDFT_recursive(X=data, scale='n')
    aae(reference_output_scipy, computed_output_recursive, decimal=10)
    scale='sqrtn'
    data = np.random.rand(2**7)+1j*np.random.rand(2**7)
    reference_output_scipy = sp.fft.ifft(x=data, norm='ortho')
    computed_output_recursive = mca.IDFT_recursive(X=data, scale='sqrtn')
    aae(reference_output_scipy, computed_output_recursive, decimal=10)


def test_DFT_parseval_theorem():
    'energy is the same in time/space and Fourier domains'
    np.random.seed(111)
    data = np.random.rand(2**4)
    X_dumb = mca.DFT_dumb(x=data, scale='sqrtn')
    X_recursive = mca.DFT_recursive(x=data, scale='sqrtn')
    energy_data = np.dot(a=data, b=data)
    energy_X_dumb = np.dot(a=X_dumb, b=np.conj(X_dumb)).real
    energy_X_recursive = np.dot(a=X_recursive, b=np.conj(X_recursive)).real
    aae(energy_data, energy_X_dumb, decimal=10)
    aae(energy_data, energy_X_recursive, decimal=10)

# Householder Vector
def test_House_vector_parameter_beta():
    'verify that beta = 2/dot(v,v)'
    np.random.seed(23)
    a = np.random.rand(7)
    v, beta = mca.House_vector(x=a)
    aae(beta, 2/np.dot(v,v), decimal=10)

def test_House_vector_orthogonal_reflection():
    'verify that the resulting Householder reflection is orthogonal'
    np.random.seed(14)
    # real vector
    a = np.random.rand(7)
    v, beta = mca.House_vector(x=a)
    P = np.identity(a.size) - beta*np.outer(v,v)
    aae(np.dot(P.T,P), np.dot(P,P.T), decimal=10)
    aae(np.dot(P.T,P), np.identity(a.size), decimal=10)

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
    aae(PA1, PA2, decimal=10)
    AP1 = np.dot(A, P)
    AP2 = mca.House_matvec(A=A, v=v, beta=beta, order='AP')
    aae(AP1, AP2, decimal=10)


# Givens rotations
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
    aae(Gv[1], 0, decimal=10)
    # b = 0
    a = 10*np.random.rand()
    b = 0
    c, s = mca.Givens_rotation(a=a, b=b)
    G = np.array([[ c, s],
                  [-s, c]])
    v = np.array([a, b])
    Gv = np.dot(G.T, v)
    aae(Gv[1], 0, decimal=10)
    # |b| > |a|
    a = 7*np.random.rand()
    b = 10*np.random.rand()
    c, s = mca.Givens_rotation(a=a, b=b)
    G = np.array([[ c, s],
                  [-s, c]])
    v = np.array([a, b])
    Gv = np.dot(G.T, v)
    aae(Gv[1], 0, decimal=10)


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
    A2[[i,k],:] = mca.Givens_matvec(A=A, c=c, s=s, i=i, k=k, order='GTA')
    aae(A2, np.dot(G.T,A), decimal=10)
    # verify AG
    G = np.identity(N)
    G[i,i] = c
    G[i,k] = s
    G[k,i] = -s
    G[k,k] = c
    A2 = A.copy()
    A2[:,[i,k]] = mca.Givens_matvec(A=A, c=c, s=s, i=i, k=k, order='AG')
    aae(A2, np.dot(A,G), decimal=10)


def test_cs2rho_rho2cs():
    'verify consistency'
    np.random.seed(11)
    a = 10*np.random.rand()
    b = 10*np.random.rand()
    c, s = mca.Givens_rotation(a=a, b=b)
    rho = mca.cs2rho(c=c, s=s)
    c2, s2 = mca.rho2cs(rho=rho)
    aae(c, c2, decimal=10)
    aae(s, s2, decimal=10)


# QR factorization
def test_House_QR_Q_from_A_Q_decomposition():
    'verify the computed Q and R matrices'
    np.random.seed(7)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    A2 = A.copy()
    mca.House_QR(A2)
    Q = mca.Q_from_A(A=A2)
    R = np.triu(A2)
    aae(A, np.dot(Q, R[:N,:]), decimal=10)


def test_House_QR_Q_from_A_Q_orthogonal():
    'verify the orthogonality of the computed Q'
    np.random.seed(78)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    mca.House_QR(A)
    Q = mca.Q_from_A(A=A)
    aae(np.identity(N), np.dot(Q.T,Q), decimal=10)


def test_House_QR_Q_from_A_R_Cholesky():
    'verify that R is the transpose of the Cholesky factor'
    np.random.seed(8)
    M = 6
    N = 5
    A = np.random.rand(M,N)
    ATA = np.dot(A.T,A)
    mca.House_QR(A)
    R = np.triu(A)[:N,:].T
    aae(ATA, np.dot(R,R.T), decimal=10)
