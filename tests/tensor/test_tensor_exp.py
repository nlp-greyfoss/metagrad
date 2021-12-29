import numpy as np

from core.tensor import Tensor


def test_simple_exp():
    x = Tensor(2, requires_grad=True)
    z = x.exp()  # e^2

    np.testing.assert_array_almost_equal(z.data, np.exp(2))

    z.backward()

    np.testing.assert_array_almost_equal(x.grad.data, np.exp(2))


def test_array_exp():
    x = Tensor([1, 2, 3], requires_grad=True)
    z = x.exp()

    np.testing.assert_array_almost_equal(z.data, np.exp([1, 2, 3]))

    z.backward([1, 1, 1])

    np.testing.assert_array_almost_equal(x.grad.data, np.exp([1, 2, 3]))