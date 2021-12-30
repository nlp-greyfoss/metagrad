from metagrad.tensor import Tensor
import numpy as np


def test_get_by_index():
    x = Tensor([1, 2, 3, 4, 5, 6, 7], requires_grad=True)
    z = x[2]

    assert z.data == 3
    z.backward()

    assert x.grad.data.tolist() == [0, 0, 1, 0, 0, 0, 0]


def test_slice():
    x = Tensor([1, 2, 3, 4, 5, 6, 7], requires_grad=True)
    z = x[2:4]

    assert z.data.tolist() == [3, 4]
    z.backward([1, 1])

    assert x.grad.data.tolist() == [0, 0, 1, 1, 0, 0, 0]


def test_matrix_slice():
    a = np.array([[1., 1., 8., 9., 1.],
                  [4., 5., 9., 9., 8.],
                  [8., 6., 9., 7., 9.],
                  [8., 6., 1., 9., 8.]])

    x = Tensor(a, requires_grad=True)
    z = x[1:3, 2:4]

    assert z.data.tolist() == [[9, 9], [9, 7]]
    z.backward([[1, 1], [1, 1]])

    # 总共有6个9
    np.testing.assert_array_almost_equal(x.grad.data, [[0, 0, 0, 0, 0],
                                                       [0, 0, 1, 1, 0],
                                                       [0, 0, 1, 1, 0],
                                                       [0, 0, 0, 0, 0]])
