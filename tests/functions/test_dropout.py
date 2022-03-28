import numpy as np
import metagrad.functions as F
from metagrad.tensor import Tensor


def test_forward():
    x = np.random.randn(100, 10)
    y = F.dropout(Tensor(x), p=0)
    assert np.allclose(y.data, x)

    y = F.dropout(x, p=0.5, training=False)

    assert np.allclose(y.data, x)


def test_backward():
    x = np.ones((100, 10))
    x = Tensor(x, requires_grad=True)
    y = F.dropout(x, p=0.75)  # 经过Dropout缩放后，非零元素变成了4
    y.sum().backward()

    # 反向传播时，非零元素的梯度应该是1/0.25=4
    assert np.allclose(x.grad.data, y.data)
