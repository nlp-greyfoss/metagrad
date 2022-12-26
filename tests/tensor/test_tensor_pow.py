import numpy as np

from metagrad.tensor import Tensor


def test_simple_pow():
    x = Tensor(2, requires_grad=True)
    y = 2
    z = x ** y

    assert z.data == 4

    z.backward()

    assert x.grad == 4


def test_array_pow():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = 3
    z = x ** y

    assert z.data.tolist() == [1, 8, 27]

    z.backward(np.array([1, 1, 1]))

    assert x.grad.tolist() == [3, 12, 27]
