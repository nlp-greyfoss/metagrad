import numpy as np

from metagrad.tensor import Tensor


def test_simple_div():
    '''
    测试简单的除法
    '''
    x = Tensor(1, requires_grad=True)
    y = Tensor(2, requires_grad=True)
    z = x / y
    z.backward()
    assert x.grad.data == 0.5
    assert y.grad.data == -0.25


def test_array_div():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([2, 4, 6], requires_grad=True)

    z = x / y

    assert z.data.tolist() == [0.5, 0.5, 0.5]
    assert x.data.tolist() == [1, 2, 3]

    z.backward(Tensor([1, 1, 1]))

    np.testing.assert_array_almost_equal(x.grad.data, [0.5, 0.25, 1 / 6])
    np.testing.assert_array_almost_equal(y.grad.data, [-0.25, -1 / 8, -1 / 12])

    x /= 0.1
    assert x.grad is None
    assert x.data.tolist() == [10, 20, 30]


def test_broadcast_div():
    x = Tensor([[1, 1, 1], [2, 2, 2]], requires_grad=True)  # (2, 3)
    y = Tensor([4, 4, 4], requires_grad=True)  # (3, )

    z = x / y  # (2,3) * (3,) => (2,3) * (2,3) -> (2,3)

    assert z.data.tolist() == [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]

    z.backward(Tensor([[1, 1, 1, ], [1, 1, 1]]))

    assert x.grad.data.tolist() == [[1/4, 1/4, 1/4], [1/4, 1/4, 1/4]]
    assert y.grad.data.tolist() == [-3/16, -3/16, -3/16]
