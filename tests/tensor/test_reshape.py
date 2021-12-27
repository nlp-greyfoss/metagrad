import numpy as np

from core.tensor import Tensor


def test_reshape():
    x = Tensor(np.arange(9), requires_grad=True)
    z = x.reshape((3, 3))
    z.backward(np.ones((3, 3)))

    assert x.grad.data.tolist() == np.ones_like(x.data).tolist()


def test_matrix_reshape():
    x = Tensor(np.arange(12).reshape(2, 6), requires_grad=True)
    z = x.reshape((4, 3))

    z.backward(np.ones((4, 3)))

    assert x.grad.data.tolist() == np.ones_like(x.data).tolist()
