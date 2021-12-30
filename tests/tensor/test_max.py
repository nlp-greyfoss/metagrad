from metagrad.tensor import Tensor
import numpy as np


def test_simple_max():
    x = Tensor([1, 2, 3, 6, 7, 9, 2], requires_grad=True)
    z = x.max()

    assert z.data == [9]
    z.backward()

    assert x.grad.data.tolist() == [0, 0, 0, 0, 0, 1, 0]


def test_simple_max2():
    x = Tensor([1, 2, 3, 9, 7, 9, 2], requires_grad=True)
    z = x.max()

    assert z.data == [9]  # 最大值还是9
    z.backward()

    # 但是有两个最大值，所以梯度被均分了
    assert x.grad.data.tolist() == [0, 0, 0, 0.5, 0, 0.5, 0]


def test_matrix_max():
    a = np.array([[1., 1., 8., 9., 1.],
                  [4., 5., 9., 9., 8.],
                  [8., 6., 9., 7., 9.],
                  [8., 6., 1., 9., 8.]])

    x = Tensor(a, requires_grad=True)
    z = x.max()

    assert z.data == [9]  # 最大值是9
    z.backward()

    # 总共有6个9
    np.testing.assert_array_almost_equal(x.grad.data, [[0, 0, 0, 1 / 6, 0],
                                                       [0, 0, 1 / 6, 1 / 6, 0],
                                                       [0, 0, 1 / 6, 0, 1 / 6],
                                                       [0, 0, 0, 1 / 6, 0]])


def test_matrix_max2():
    a = np.array([[1., 1., 8., 9., 1.],
                  [4., 5., 9., 9., 8.],
                  [8., 6., 9., 7., 9.],
                  [8., 6., 1., 9., 8.]])

    x = Tensor(a, requires_grad=True)
    z = x.max(axis=0)  # [8, 6, 9, 9, 9]

    assert z.data.tolist() == [8, 6, 9, 9, 9]
    z.backward([1, 1, 1, 1, 1])

    grad = [[0., 0., 0., 1 / 3, 0.],
            [0., 0., 0.5, 1 / 3, 0.],
            [0.5, 0.5, 0.5, 0, 1],
            [0.5, 0.5, 0., 1 / 3, 0.]]

    np.testing.assert_array_almost_equal(x.grad.data, np.array(grad))
