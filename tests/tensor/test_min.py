from metagrad.tensor import Tensor
import numpy as np
import torch


def test_simple_min():
    x = Tensor([1, 2, 3, 6, 7, 9, 2], requires_grad=True)
    z = x.min()

    assert z.data == [1]
    z.backward()

    assert x.grad.data.tolist() == [1, 0, 0, 0, 0, 0, 0]


def test_simple_min2():
    x = Tensor([1, 2, 3, 9, 7, 9, 1], requires_grad=True)
    z = x.min()

    assert z.data == [1]  # 最小值还是1
    z.backward()

    # 但是有两个最小值，所以梯度被均分了
    assert x.grad.data.tolist() == [0.5, 0, 0, 0, 0, 0, 0.5]


def test_matrix_min():
    a = np.array([[1., 1., 8., 9., 1.],
                  [4., 5., 9., 9., 8.],
                  [8., 6., 9., 7., 9.],
                  [8., 6., 1., 9., 8.]])

    x = Tensor(a, requires_grad=True)
    z = x.min()

    assert z.data == [1]  # 最小值是1
    z.backward()

    # 总共有4个1
    np.testing.assert_array_almost_equal(x.grad.data, [[1 / 4, 1 / 4, 0, 0, 1 / 4],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 1 / 4, 0, 0]])


def test_matrix_min2():
    a = np.array([[1., 1., 8., 9., 1.],
                  [4., 5., 9., 9., 8.],
                  [8., 6., 9., 7., 9.],
                  [8., 6., 1., 9., 8.]])

    x = Tensor(a, requires_grad=True)
    z = x.min(0)  # [1, 1, 1, 7, 1]

    assert z.data.tolist() == [1, 1, 1, 7, 1]
    z.backward([1, 1, 1, 1, 1])

    grad = [[1., 1., 0., 0., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0.]]

    np.testing.assert_array_almost_equal(x.grad.data, np.array(grad))


def test_matrix_with_axis():
    a = np.arange(24).reshape(2, 3, 4)

    mx = Tensor(a, requires_grad=True)
    #    Tensor([[[ 0  1  2  3]
    #             [ 4  5  6  7]
    #             [ 8  9 10 11]]
    #
    #            [[12 13 14 15]
    #             [16 17 18 19]
    #             [20 21 22 23]]]

    my = mx.min(1)

    tx = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    ty = tx.min(1)

    assert np.allclose(my.data, ty.values.data)

    my.sum().backward()
    ty.values.sum().backward()

    np.testing.assert_array_almost_equal(tx.grad.data, mx.grad.data)


def test_matrix_with_negative_axis():
    a = np.arange(16).reshape(2, 2, 4)

    mx = Tensor(a, requires_grad=True)

    my = mx.min(-2)

    tx = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    ty = tx.min(-2)

    assert np.allclose(my.data, ty.values.data)

    my.sum().backward()
    ty.values.sum().backward()

    np.testing.assert_array_almost_equal(tx.grad.data, mx.grad.data)
