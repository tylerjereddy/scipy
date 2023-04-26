import numpy as np
from numpy.testing import assert_equal
import pytest

from scipy.conftest import array_api_compatible
from scipy._lib._array_api import (
    SCIPY_ARRAY_API, array_namespace, asarray, asarray_namespace,
    to_numpy
)


if not SCIPY_ARRAY_API:
    pytest.skip(
        "Array API test; set environment variable array_api_dispatch=1 to run it",
        allow_module_level=True
    )


def test_array_namespace():
    x, y = [0, 1, 2], np.arange(3)
    xp = array_namespace(x, y)
    assert xp.__name__ == 'array_api_compat.numpy'


@array_api_compatible
def test_asarray(xp):
    x, y = asarray([0, 1, 2], xp=xp), asarray(np.arange(3), xp=xp)
    ref = np.array([0, 1, 2])
    assert_equal(x, ref)
    assert_equal(y, ref)


def test_asarray_namespace():
    x, y = [0, 1, 2], np.arange(3)
    x, y, xp_ = asarray_namespace(x, y)
    assert xp_.__name__ == 'array_api_compat.numpy'
    ref = np.array([0, 1, 2])
    assert_equal(x, ref)
    assert_equal(y, ref)
    assert type(x) == type(y)


@array_api_compatible
def test_to_numpy(xp):
    x = xp.asarray([0, 1, 2])
    x = to_numpy(x, xp=xp)
    assert isinstance(x, np.ndarray)


@pytest.mark.filterwarnings("ignore: the matrix subclass")
def test_raises():
    msg = "'numpy.ma.MaskedArray' are not supported"
    with pytest.raises(TypeError, match=msg):
        array_namespace(np.ma.array(1), np.array(1))

    msg = "'numpy.matrix' are not supported"
    with pytest.raises(TypeError, match=msg):
        array_namespace(np.array(1), np.matrix(1))

