from metagrad.tensor import Tensor
import numpy as np


def test_get_by_index():
    x = Tensor([1, 2, 3, 4, 5, 6, 7], requires_grad=True)
    z = x[Tensor(2)]

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

    assert x[0, 0].item() == 1


def test_boolean_indexing():
    '''测试boolean索引操作'''
    x = Tensor([1, 2, 3, 4, 5, 6, 7], requires_grad=True)
    z = x[x < 5]

    assert z.data.tolist() == [1., 2., 3., 4.]
    z.sum().backward()

    assert x.grad.data.tolist() == [1, 1, 1, 1, 0, 0, 0]


def test_integer_indexing():
    x = Tensor(np.arange(35).reshape(5, 7), requires_grad=True)
    # Tensor([[(0)  1  2  3  4  5  6]
    #         [7  8  9 10 11 12 13]
    #         [14 (15) 16 17 18 19 20]
    #         [21 22 23 24 25 26 27]
    #         [28 29 (30) 31 32 33 34]], requires_grad = True)
    #

    # ! z = x[Tensor([0, 2, 4]), Tensor([0, 1, 2])] 暂不支持元组Tensor作为索引
    z = x[np.array([0, 2, 4]), np.array([0, 1, 2])]  # x[0,0] x[2,1] x[4,2]

    assert z.data.tolist() == [0, 15, 30]

    z.sum().backward()

    assert x.grad.data.tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]


def test_array_indexing():
    a = np.random.randn(32, 56)
    mask = a > 0
    print(mask.shape)

    b = np.random.randn(32, 56, 47)
    print(b[mask].shape)

    x = Tensor(b, requires_grad=True)

    mask = Tensor(mask)
    y = x[mask]
    print(y.shape)
