from metagrad.tensor import Tensor
import numpy as np


def test_right_add_constant():
    x = Tensor(1, requires_grad=True)
    z = x + 2
    z.backward()
    assert x.grad == 1.0


def test_left_add_constant():
    x = Tensor(1, requires_grad=True)
    z = 2 + x
    z.backward()
    assert x.grad == 1.0


def test_array_add():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)

    z = x + y
    assert z.tolist() == [5., 7., 9.]

    # 如果
    z.backward(np.array([1, 1, 1]))

    assert x.grad.tolist() == [1, 1, 1]
    assert y.grad.tolist() == [1, 1, 1]

    x += 1
    assert x.tolist() == [2, 3, 4]


def test_broadcast_add():
    """
    测试当发生广播时，我们的代码还能表现正常吗。
    对于 z = x + y
    如果x.shape == y.shape，那么就像上面的例子一样，没什么问题。
    如果x.shape == (2,3)  y.shape == (3,) 那么，根据广播，先会在y左边插入一个维度1，变成 -> y.shape == (1,3)
        接着，在第0个维度上进行复制，使得新的维度 y.shape == (2,3)
    这样的话，对x求梯度时，梯度要和x的shape保持一致；对y求梯度时，也要和y的shape保持一致。
    """
    x = Tensor(np.random.randn(2, 3), requires_grad=True)  # (2,3)
    y = Tensor(np.random.randn(3), requires_grad=True)  # (3,)

    z = x + y  # (2,3)

    z.backward(np.ones_like(x.data))  # grad.shape == z.shape

    assert x.grad.tolist() == np.ones_like(x.data).tolist()
    assert y.grad.tolist() == np.array([2, 2, 2]).tolist()


def test_broadcast_add2():
    x = Tensor(np.random.randn(2, 3), requires_grad=True)  # (2,3)
    y = Tensor(np.random.randn(1, 3), requires_grad=True)  # (1,3)

    z = x + y  # (2,3)

    z.backward(np.ones_like(x.data))  # grad.shape == z.shape

    assert x.grad.tolist() == np.ones_like(x.data).tolist()
    assert y.grad.tolist() == (np.ones_like(y.data) * 2).tolist()
