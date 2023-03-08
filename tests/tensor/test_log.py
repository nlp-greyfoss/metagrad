import math

import numpy as np

from metagrad.tensor import Tensor


def test_simple_log():
    x = Tensor(10, requires_grad=True)
    z = x.log()

    np.testing.assert_array_almost_equal(z.data, math.log(10))

    z.backward()

    np.testing.assert_array_almost_equal(x.grad.tolist(), 0.1)


def test_array_log():
    x = Tensor([1, 2, 3], requires_grad=True)
    z = x.log()

    np.testing.assert_array_almost_equal(z.data, np.log([1, 2, 3]))

    z.backward(np.array([1, 1, 1]))

    np.testing.assert_array_almost_equal(x.grad.tolist(), [1, 0.5, 1 / 3])
