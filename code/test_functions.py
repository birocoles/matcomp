import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import functions as fcs

### dot_product

def test_dot_product_not_1D_arrays():
    'check input not 1D arrays'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        fcs.dot_product(vector_1, vector_2)


def test_dot_product_different_sizes():
    'check input with different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        fcs.dot_product(vector_1, vector_2)


def test_dot_product_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output = fcs.dot_product(vector_1, vector_2)
    aae(reference_output, computed_output, decimal=10)
