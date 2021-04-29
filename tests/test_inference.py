import pytest

import deepinterpolation.loss_collection as lc


@pytest.mark.parametrize(
    "x, expected",
    [(2, 4), (3, 9), (1.234, 1.234**2)])
def test_dummy_function(x, expected):
    assert expected == lc.dummy_function(x)
