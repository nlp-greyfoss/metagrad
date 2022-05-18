import numpy as np

from metagrad.tensor import Tensor


def test_simple_sum():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x.sum()

    assert y.data == 6

    y.backward()

    assert x.grad.data.tolist() == [1, 1, 1]


def test_sum_with_grad():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x.sum()

    y.backward(Tensor(3))

    assert x.grad.data.tolist() == [3, 3, 3]


def test_matrix_sum():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    y = x.sum()
    assert y.data == 21

    y.backward()

    assert x.grad.data.tolist() == np.ones_like(x.data).tolist()


def test_matrix_with_axis():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    y = x.sum(0)  # keepdims = False

    assert y.shape == (3,)
    assert y.data.tolist() == [5, 7, 9]

    y.backward([1, 1, 1])

    assert x.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_matrix_with_keepdims():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    y = x.sum(axis=0, keepdims=True)  # keepdims = True
    assert y.shape == (1, 3)
    assert y.data.tolist() == [[5, 7, 9]]
    y.backward([1, 1, 1])

    assert x.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_complex_matrix_with_axis():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)  # (2,3,4)
    y = x.sum((0, 1))  # keepdims = False

    assert y.shape == (4,)
    assert y.data.tolist() == [60, 66, 72, 78]

    y.backward([2, 2, 2, 2])

    assert x.grad.data.tolist() == [[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
                                    [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]]


def test_sum_with_negative_axis():
    x = Tensor(np.arange(16).reshape(2, 2, 4), requires_grad=True)  # (2,2,4)
    y = x.sum(-2)  # keepdims = False

    assert y.shape == (2, 4)
    assert y.data.tolist() == [[4, 6, 8, 10], [20, 22, 24, 26]]

    y.sum().backward()

    assert x.grad.data.tolist() == [[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]]
