import numpy as np

from metagrad.tensor import Tensor


def test_simple_clip():
    x = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    # 无参数，输出不变，所有元素都有梯度
    y = x.clip()

    assert np.allclose(y.numpy(), x.numpy())

    y.backward(np.ones_like(x.numpy()))
    assert x.grad.data.tolist() == [1, 1, 1, 1, 1]


def test_clip_with_min():
    x = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    # 最小值为3
    y = x.clip(x_min=3)

    assert y.numpy().tolist() == [3, 3, 3, 4, 5]

    y.backward(np.ones_like(x.numpy()))
    assert x.grad.data.tolist() == [0, 0, 1, 1, 1]


def test_clip_with_max():
    x = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    # 最大值为3
    y = x.clip(x_max=3)

    assert y.numpy().tolist() == [1, 2, 3, 3, 3]

    y.backward(np.ones_like(x.numpy()))
    assert x.grad.data.tolist() == [1, 1, 1, 0, 0]


def test_clip():
    x = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], requires_grad=True)
    # 最小值为3，最大值为6
    y = x.clip(x_min=3, x_max=6)

    assert y.numpy().tolist() == [3, 3, 3, 4, 5, 6, 6, 6, 6]

    y.backward(np.ones_like(x.numpy()))
    assert x.grad.data.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
