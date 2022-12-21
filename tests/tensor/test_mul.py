import numpy as np

from metagrad.tensor import Tensor


def test_simple_mul():
    '''
    测试简单的乘法
    '''
    x = Tensor(1, requires_grad=True)
    y = 2
    z = x * y
    z.backward()
    assert x.grad == 2.0


def test_array_mul():
    '''
    测试两个同shape的向量乘法
    '''
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)

    z = x * y

    # 对应元素相乘
    assert z.data.tolist() == [4, 10, 18]

    z.backward(np.array([1, 1, 1]))

    assert x.grad.tolist() == y.data.tolist()
    assert y.grad.tolist() == x.data.tolist()

    x *= 0.1
    assert x.grad is None

    # assert [0.10000000149011612, 0.20000000298023224, 0.30000001192092896] == [0.1, 0.2, 0.3]
    # assert x.data.tolist() == [0.1, 0.2, 0.3]
    # 需要用近似相等来判断
    np.testing.assert_array_almost_equal(x.data, [0.1, 0.2, 0.3])


def test_broadcast_mul():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    y = Tensor([7, 8, 9], requires_grad=True)  # (3, )

    z = x * y  # (2,3) * (3,) => (2,3) * (2,3) -> (2,3)

    assert z.data.tolist() == [[7, 16, 27], [28, 40, 54]]

    z.backward(np.array([[1, 1, 1, ], [1, 1, 1]]))

    assert x.grad.tolist() == [[7, 8, 9], [7, 8, 9]]
    assert y.grad.tolist() == [5, 7, 9]


def test_broadcast_mul2():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    y = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

    z = x * y  # (2,3) * (1,3) => (2,3) * (2,3) -> (2,3)

    assert z.data.tolist() == [[7, 16, 27], [28, 40, 54]]

    z.backward(np.array([[1, 1, 1, ], [1, 1, 1]]))

    assert x.grad.tolist() == [[7, 8, 9], [7, 8, 9]]
    assert y.grad.tolist() == [[5, 7, 9]]
