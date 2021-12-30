import numpy as np

from metagrad.tensor import Tensor


def test_transpose():
    x = Tensor(np.arange(6).reshape((2, 3)), requires_grad=True)
    z = x.T

    assert z.data.shape == (3, 2)
    z.backward(np.ones((3, 2)))

    assert x.grad.data.tolist() == np.ones_like(x.data).tolist()


def test_matrix_transpose():
    x = Tensor(np.arange(12).reshape((2, 6, 1)), requires_grad=True)
    z = x.transpose((0, 2, 1))

    assert z.data.shape == (2, 1, 6)

    z.backward(np.ones((2, 1, 6)))

    assert x.grad.data.tolist() == np.ones_like(x.data).tolist()
