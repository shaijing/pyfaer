import pytest
import pyfaer


def test_sum_as_string():
    assert pyfaer.sum_as_string(1, 1) == "2"
