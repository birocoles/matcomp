import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import functions as fcs

### dot_product

def test_dot_not_1D_arrays():
    'check input not 1D arrays'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        fcs.dot_dumb(vector_1, vector_2)
        fcs.dot_numpy(vector_1, vector_2)
        fcs.dot_numba(vector_1, vector_2)


def test_dot_different_sizes():
    'check input with different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        fcs.dot_dumb(vector_1, vector_2)
        fcs.dot_numpy(vector_1, vector_2)
        fcs.dot_numba(vector_1, vector_2)


def test_dot_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output_dumb = fcs.dot_dumb(vector_1, vector_2)
    computed_output_numpy = fcs.dot_numpy(vector_1, vector_2)
    computed_output_numba = fcs.dot_numba(vector_1, vector_2)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_dot_compare_numpy_dot():
    'compare with numpy.dot'
    np.random.seed = 41
    vector_1 = np.random.rand(13)
    vector_2 = np.random.rand(13)
    reference_output_numpy = np.dot(vector_1, vector_2)
    computed_output_dumb = fcs.dot_dumb(vector_1, vector_2)
    computed_output_numpy = fcs.dot_numpy(vector_1, vector_2)
    computed_output_numba = fcs.dot_numba(vector_1, vector_2)
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
    aae(reference_output_numpy, computed_output_numpy, decimal=10)
    aae(reference_output_numpy, computed_output_numba, decimal=10)


def test_dot_verify_complex_input():
    'verify results produced with complex inputs'
    # first input complex
    np.random.seed = 3
    vector_1 = np.random.rand(13) + np.random.rand(13)*1j
    vector_2 = np.random.rand(13)
    computed_output = fcs.dot_numpy(vector_1, vector_2)
    expected_result_real = np.sum(vector_1.real*vector_2.real)
    expected_result_real -= np.sum(vector_1.imag*vector_2.imag)
    expected_result_imag = np.sum(vector_1.real*vector_2.imag)
    expected_result_imag += np.sum(vector_1.imag*vector_2.real)
    expected_result = expected_result_real + expected_result_imag*1j
    aae(computed_output, expected_result, decimal=10)


# Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT)

def test_DFT_invalid_scalar():
    'check error for invalid scaling factor'
    data = np.zeros(10)
    invalid_scales = [data.size, np.sqrt(data.size), 'n', 'srqtn']
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            fcs.DFT_dumb(x=data, scale=invalid_scale)


def test_IDFT_invalid_scalar():
    'check error for invalid scaling factor'
    data = np.zeros(10)
    invalid_scales = [data.size, np.sqrt(data.size), 'n', 'srqtn']
    with pytest.raises(AssertionError):
        for invalid_scale in invalid_scales:
            fcs.IDFT_dumb(x=data, scale=invalid_scale)


def test_DFT_compare_numpy_fft_fft():
    'compare with numpy.fft.fft'
    np.random.seed = 56
    # unscaled DFT
    data = np.random.rand(100)
    reference_output_numpy = np.fft.fft(a=data, norm=None)
    computed_output_dumb = fcs.DFT_dumb(x=data, scale=None)
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
    # scaled DFT
    data = np.random.rand(100)
    reference_output_numpy = np.fft.fft(a=data, norm='ortho')
    computed_output_dumb = fcs.DFT_dumb(x=data, scale='sqrtn')
    aae(reference_output_numpy, computed_output_dumb, decimal=10)


def test_IDFT_compare_numpy_fft_ifft():
    'compare with numpy.fft.ifft'
    np.random.seed = 4
    # unscaled DFT
    data = np.random.rand(100)+1j*np.random.rand(100)
    reference_output_numpy = np.fft.ifft(a=data, norm=None)
    computed_output_dumb = fcs.IDFT_dumb(x=data, scale='N')
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
    # scaled DFT
    data = np.random.rand(100)+1j*np.random.rand(100)
    reference_output_numpy = np.fft.ifft(a=data, norm='ortho')
    computed_output_dumb = fcs.IDFT_dumb(x=data, scale='sqrtn')
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
