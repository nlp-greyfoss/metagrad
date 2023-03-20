import numpy as np

from metagrad.tensor import Tensor


def test_simple_exp():
    x = Tensor(2, requires_grad=True)
    z = x.exp()  # e^2

    np.testing.assert_array_almost_equal(z.data, np.exp(2))

    z.backward()

    np.testing.assert_array_almost_equal(x.grad, np.exp(2))


def test_exp():
    x = Tensor(-2, requires_grad=True)
    z = x.exp()

    z.backward(np.array(2))

    np.testing.assert_array_almost_equal(x.grad, 2 * np.exp(-2))


def test_array_exp():
    x = Tensor([1, 2, 3], requires_grad=True)
    z = x.exp()

    np.testing.assert_array_almost_equal(z.data, np.exp([1, 2, 3]))

    z.backward(np.array([1, 1, 1]))

    np.testing.assert_array_almost_equal(x.grad, np.exp([1, 2, 3]))
