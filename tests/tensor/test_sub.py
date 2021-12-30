import numpy as np

from metagrad.tensor import Tensor


def test_simple_sub():
    x = Tensor(1, requires_grad=True)
    y = Tensor(2, requires_grad=True)
    z = x - y
    z.backward()
    assert x.grad.data == 1.0
    assert y.grad.data == -1.0


def test_array_sub():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)

    z = x - y
    assert z.data.tolist() == [-3., -3., -3.]

    z.backward(Tensor([1, 1, 1]))

    assert x.grad.data.tolist() == [1, 1, 1]
    assert y.grad.data.tolist() == [-1, -1, -1]

    x -= 0.1
    assert x.grad is None
    np.testing.assert_array_almost_equal(x.data, [0.9, 1.9, 2.9])


def test_broadcast_sub():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    y = Tensor([7, 8, 9], requires_grad=True)  # (3, )

    z = x - y  # shape (2, 3)
    assert z.data.tolist() == [[-6, -6, -6], [-3, -3, -3]]

    z.backward(Tensor(np.ones_like(x.data)))

    assert x.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert y.grad.data.tolist() == [-2, -2, -2]
