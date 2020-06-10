import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import functions as fcs

### dot_product

def test_dot_real_not_1D_arrays():
    'fail due to input that is not 1D array'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        fcs.dot_real_dumb(vector_1, vector_2)
        fcs.dot_real_numpy(vector_1, vector_2)
        fcs.dot_real_numba(vector_1, vector_2)
        fcs.dot_real_parallel(vector_1, vector_2)


def test_dot_real_different_sizes():
    'fail due to inputs having different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        fcs.dot_real_dumb(vector_1, vector_2)
        fcs.dot_real_numpy(vector_1, vector_2)
        fcs.dot_real_numba(vector_1, vector_2)
        fcs.dot_real_parallel(vector_1, vector_2)


def test_dot_real_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output_dumb = fcs.dot_real_dumb(vector_1, vector_2)
    computed_output_numpy = fcs.dot_real_numpy(vector_1, vector_2)
    computed_output_numba = fcs.dot_real_numba(vector_1, vector_2)
    computed_output_parallel = fcs.dot_real_parallel(vector_1, vector_2)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)
    aae(reference_output, computed_output_parallel, decimal=10)


def test_dot_real_compare_numpy_dot():
    'compare with numpy.dot'
    np.random.seed = 41
    vector_1 = np.random.rand(13)
    vector_2 = np.random.rand(13)
    reference_output_numpy = np.dot(vector_1, vector_2)
    computed_output_dumb = fcs.dot_real_dumb(vector_1, vector_2)
    computed_output_numpy = fcs.dot_real_numpy(vector_1, vector_2)
    computed_output_numba = fcs.dot_real_numba(vector_1, vector_2)
    computed_output_parallel = fcs.dot_real_parallel(vector_1, vector_2)
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
    aae(reference_output_numpy, computed_output_numpy, decimal=10)
    aae(reference_output_numpy, computed_output_numba, decimal=10)
    aae(reference_output_numpy, computed_output_parallel, decimal=10)


def test_dot_real_commutativity():
    'verify commutativity'
    np.random.seed = 19
    a = np.random.rand(15)
    b = np.random.rand(15)
    # a dot b = b dot a
    output_ab_dumb = fcs.dot_real_dumb(a, b)
    output_ba_dumb = fcs.dot_real_dumb(b, a)
    output_ab_numpy = fcs.dot_real_numpy(a, b)
    output_ba_numpy = fcs.dot_real_numpy(b, a)
    output_ab_numba = fcs.dot_real_numba(a, b)
    output_ba_numba = fcs.dot_real_numba(b, a)
    output_ab_parallel = fcs.dot_real_parallel(a, b)
    output_ba_parallel = fcs.dot_real_parallel(b, a)
    aae(output_ab_dumb, output_ba_dumb, decimal=10)
    aae(output_ab_numpy, output_ba_numpy, decimal=10)
    aae(output_ab_numba, output_ba_numba, decimal=10)
    aae(output_ab_parallel, output_ba_parallel, decimal=10)


def test_dot_real_distributivity():
    'verify distributivity over sum'
    np.random.seed = 19
    a = np.random.rand(15)
    b = np.random.rand(15)
    c = np.random.rand(15)
    # a dot (b + c) = (a dot b) + (a dot c)
    output_a_bc_dumb = fcs.dot_real_dumb(a, b + c)
    output_ab_ac_dumb = fcs.dot_real_dumb(a, b) + fcs.dot_real_dumb(a, c)
    output_a_bc_numpy = fcs.dot_real_numpy(a, b + c)
    output_ab_ac_numpy = fcs.dot_real_numpy(a, b) + fcs.dot_real_numpy(a, c)
    output_a_bc_numba = fcs.dot_real_numba(a, b + c)
    output_ab_ac_numba = fcs.dot_real_numba(a, b) + fcs.dot_real_numba(a, c)
    output_a_bc_parallel = fcs.dot_real_parallel(a, b + c)
    output_ab_ac_parallel = fcs.dot_real_parallel(a, b) + fcs.dot_real_parallel(a, c)
    aae(output_a_bc_dumb, output_ab_ac_dumb, decimal=10)
    aae(output_a_bc_numpy, output_ab_ac_numpy, decimal=10)
    aae(output_a_bc_numba, output_ab_ac_numba, decimal=10)
    aae(output_a_bc_parallel, output_ab_ac_parallel, decimal=10)


def test_dot_real_scalar_multiplication():
    'verify scalar multiplication property'
    np.random.seed = 8
    a = np.random.rand(15)
    b = np.random.rand(15)
    c1 = 5.6
    c2 = 9.1
    # (c1 a) dot (c2 b) = c1c2 (a dot b)
    output_c1a_c2b_dumb = fcs.dot_real_dumb(c1*a, c2*b)
    output_c1c2_ab_dumb = c1*c2*fcs.dot_real_dumb(a, b)
    output_c1a_c2b_numpy = fcs.dot_real_numpy(c1*a, c2*b)
    output_c1c2_ab_numpy = c1*c2*fcs.dot_real_numpy(a, b)
    output_c1a_c2b_numba = fcs.dot_real_numba(c1*a, c2*b)
    output_c1c2_ab_numba = c1*c2*fcs.dot_real_numba(a, b)
    output_c1a_c2b_parallel = fcs.dot_real_parallel(c1*a, c2*b)
    output_c1c2_ab_parallel = c1*c2*fcs.dot_real_parallel(a, b)
    aae(output_c1a_c2b_dumb, output_c1c2_ab_dumb, decimal=10)
    aae(output_c1a_c2b_numpy, output_c1c2_ab_numpy, decimal=10)
    aae(output_c1a_c2b_numba, output_c1c2_ab_numba, decimal=10)
    aae(output_c1a_c2b_parallel, output_c1c2_ab_parallel, decimal=10)


def test_dot_complex_functions_compare_numpy_dot():
    'compare dot_complex_dumb, numpy and numba with numpy.dot'
    # first input complex
    np.random.seed = 3
    vector_1 = np.random.rand(13) + np.random.rand(13)*1j
    vector_2 = np.random.rand(13) + np.random.rand(13)*1j
    output_dumb = fcs.dot_complex_dumb(vector_1, vector_2)
    output_numpy = fcs.dot_complex_numpy(vector_1, vector_2)
    output_numba = fcs.dot_complex_numba(vector_1, vector_2)
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numpy, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)


def test_dot_complex_compare_numpy_dot():
    'compare dot_complex with numpy.dot'
    # first input complex
    np.random.seed = 78
    vector_1 = np.random.rand(10) + np.random.rand(10)*1j
    vector_2 = np.random.rand(10) + np.random.rand(10)*1j
    output_dumb = fcs.dot_complex(vector_1, vector_2, function='dumb')
    output_numpy = fcs.dot_complex(vector_1, vector_2, function='numpy')
    output_numba = fcs.dot_complex(vector_1, vector_2, function='numba')
    output_parallel = fcs.dot_complex(vector_1, vector_2, function='parallel')
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numpy, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)
    aae(output_parallel, output_numpy_dot, decimal=10)


def test_dot_complex_compare_numpy_vdot():
    'compare dot_complex with numpy.vdot'
    # first input complex
    np.random.seed = 78
    vector_1 = np.random.rand(10) + np.random.rand(10)*1j
    vector_2 = np.random.rand(10) + np.random.rand(10)*1j
    output_dumb = fcs.dot_complex(vector_1, vector_2,
                                  conjugate=True, function='dumb')
    output_numpy = fcs.dot_complex(vector_1, vector_2,
                                   conjugate=True, function='numpy')
    output_numba = fcs.dot_complex(vector_1, vector_2,
                                   conjugate=True, function='numba')
    output_parallel = fcs.dot_complex(vector_1, vector_2,
                                      conjugate=True, function='parallel')
    output_numpy_dot = np.vdot(vector_1, vector_2)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numpy, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)
    aae(output_parallel, output_numpy_dot, decimal=10)


def test_dot_complex_invalid_function():
    'fail due to invalid function'
    vector_1 = np.ones(10)
    vector_2 = np.arange(10)+1.5
    with pytest.raises(ValueError):
        fcs.dot_complex(vector_1, vector_2, function='not_valid_function')


# Hadamard product

def test_hadamard_real_different_shapes():
    'fail due to inputs having different sizes'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        fcs.hadamard_real_dumb(a, B)
        fcs.hadamard_real_numpy(a, B)
        fcs.hadamard_real_numba(a, B)
        fcs.hadamard_real_parallel(a, B)


def test_hadamard_real_compare_asterisk():
    'compare hadamard_real function with * operator'
    # for vectors
    np.random.seed = 7
    input1 = np.random.rand(10)
    input2 = np.random.rand(10)
    output_dumb = fcs.hadamard_real_dumb(input1, input2)
    output_numpy = fcs.hadamard_real_numba(input1, input2)
    output_numba = fcs.hadamard_real_numba(input1, input2)
    output_parallel = fcs.hadamard_real_parallel(input1, input2)
    output_asterisk = input1*input2
    aae(output_dumb, output_asterisk, decimal=10)
    aae(output_numpy, output_asterisk, decimal=10)
    aae(output_numba, output_asterisk, decimal=10)
    aae(output_parallel, output_asterisk, decimal=10)
    # for matrices
    np.random.seed = 9
    input1 = np.random.rand(5, 7)
    input2 = np.random.rand(5, 7)
    output_dumb = fcs.hadamard_real_dumb(input1, input2)
    output_numpy = fcs.hadamard_real_numba(input1, input2)
    output_numba = fcs.hadamard_real_numba(input1, input2)
    output_parallel = fcs.hadamard_real_parallel(input1, input2)
    output_asterisk = input1*input2
    aae(output_dumb, output_asterisk, decimal=10)
    aae(output_numpy, output_asterisk, decimal=10)
    aae(output_numba, output_asterisk, decimal=10)
    aae(output_parallel, output_asterisk, decimal=10)


def test_hadamard_complex_compare_asterisk():
    'compare hadamard_complex function with * operator'
    # for matrices
    np.random.seed = 34
    input1 = np.random.rand(4, 3)
    input2 = np.random.rand(4, 3)
    output_dumb = fcs.hadamard_complex(input1, input2, function='dumb')
    output_numpy = fcs.hadamard_complex(input1, input2, function='numpy')
    output_numba = fcs.hadamard_complex(input1, input2, function='numba')
    output_parallel = fcs.hadamard_complex(input1, input2, function='parallel')
    output_asterisk = input1*input2
    aae(output_dumb, output_asterisk, decimal=10)
    aae(output_numpy, output_asterisk, decimal=10)
    aae(output_numba, output_asterisk, decimal=10)
    aae(output_parallel, output_asterisk, decimal=10)


def test_hadamard_complex_invalid_function():
    'fail due to invalid function'
    vector_1 = np.ones(8)
    vector_2 = np.arange(8)+1.5
    with pytest.raises(ValueError):
        fcs.hadamard_complex(vector_1, vector_2, function='not_valid_function')




# Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT)

# def test_DFT_invalid_scalar():
#     'check error for invalid scaling factor'
#     data = np.zeros(10)
#     invalid_scales = [data.size, np.sqrt(data.size), 'n', 'srqtn']
#     with pytest.raises(AssertionError):
#         for invalid_scale in invalid_scales:
#             fcs.DFT_dumb(x=data, scale=invalid_scale)
#
#
# def test_IDFT_invalid_scalar():
#     'check error for invalid scaling factor'
#     data = np.zeros(10)
#     invalid_scales = [data.size, np.sqrt(data.size), 'n', 'srqtn']
#     with pytest.raises(AssertionError):
#         for invalid_scale in invalid_scales:
#             fcs.IDFT_dumb(x=data, scale=invalid_scale)
#
#
# def test_DFT_compare_numpy_fft_fft():
#     'compare with numpy.fft.fft'
#     np.random.seed = 56
#     # unscaled DFT
#     data = np.random.rand(100)
#     reference_output_numpy = np.fft.fft(a=data, norm=None)
#     computed_output_dumb = fcs.DFT_dumb(x=data, scale=None)
#     aae(reference_output_numpy, computed_output_dumb, decimal=10)
#     # scaled DFT
#     data = np.random.rand(100)
#     reference_output_numpy = np.fft.fft(a=data, norm='ortho')
#     computed_output_dumb = fcs.DFT_dumb(x=data, scale='sqrtn')
#     aae(reference_output_numpy, computed_output_dumb, decimal=10)
#
#
# def test_IDFT_compare_numpy_fft_ifft():
#     'compare with numpy.fft.ifft'
#     np.random.seed = 4
#     # unscaled DFT
#     data = np.random.rand(100)+1j*np.random.rand(100)
#     reference_output_numpy = np.fft.ifft(a=data, norm=None)
#     computed_output_dumb = fcs.IDFT_dumb(x=data, scale='N')
#     aae(reference_output_numpy, computed_output_dumb, decimal=10)
#     # scaled DFT
#     data = np.random.rand(100)+1j*np.random.rand(100)
#     reference_output_numpy = np.fft.ifft(a=data, norm='ortho')
#     computed_output_dumb = fcs.IDFT_dumb(x=data, scale='sqrtn')
#     aae(reference_output_numpy, computed_output_dumb, decimal=10)
