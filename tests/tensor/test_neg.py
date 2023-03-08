import numpy as np

from metagrad.tensor import Tensor


def test_simple_neg():
    x = Tensor(2, requires_grad=True)
    z = -x  # -2

    assert z.data == -2

    z.backward()

    assert x.grad == -1


def test_array_neg():
    x = Tensor([1, 2, 3], requires_grad=True)

    z = -x

    np.testing.assert_array_equal(z.data, [-1, -2, -3])

    z.backward(np.array([1, 1, 1]))

    np.testing.assert_array_equal(x.grad, [-1, -1, -1])
