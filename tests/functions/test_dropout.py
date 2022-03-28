import numpy as np
import metagrad.functions as F
from metagrad.tensor import Tensor, eval_mode


def test_forward():
    x = np.random.randn(100, 10)
    y = F.dropout(Tensor(x), p=0)
    assert np.allclose(y.data, x)

    with eval_mode():
        y = F.dropout(x, p=0.5)

    assert np.allclose(y.data, x)


def test_backward():
    x = np.ones((100, 10))
    x = x * 5  # 每个元素值都是5
    x = Tensor(x, requires_grad=True)
    y = F.dropout(x)  # 经过Dropout缩放后，非零元素变成了10
    y.sum().backward()

    # 反向传播时，非零元素的梯度应该是2
    assert np.allclose(x.grad.data, y.data / 5)
