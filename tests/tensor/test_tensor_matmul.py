import numpy as np
import torch

from metagrad.tensor import Tensor
from torch import tensor


def test_simple_matmul():
    x = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)  # (3,2)
    y = Tensor([[2], [3]], requires_grad=True)  # (2, 1)

    z = x @ y  # (3,2) @ (2, 1) -> (3,1)

    assert z.data.tolist() == [[8], [18], [28]]

    grad = Tensor(np.ones_like(z.data))
    z.backward(grad)

    np.testing.assert_array_equal(x.grad.data, grad.data @ y.data.T)
    np.testing.assert_array_equal(y.grad.data, x.data.T @ grad.data)


def test_broadcast_matmul():
    x = Tensor(np.arange(2 * 2 * 4).reshape((2, 2, 4)), requires_grad=True)  # (2, 2, 4)
    y = Tensor(np.arange(2 * 4).reshape((4, 2)), requires_grad=True)  # (4, 2)

    z = x @ y  # (2,2,4) @ (4,2) -> (2,2,4) @ (1,4,2) => (2,2,4) @ (2,4,2)  -> (2,2,2)
    assert z.shape == (2, 2, 2)

    # 引入torch.tensor进行测试
    tx = tensor(x.data, dtype=torch.float, requires_grad=True)
    ty = tensor(y.data, dtype=torch.float, requires_grad=True)
    tz = tx @ ty

    assert z.data.tolist() == tz.data.tolist()

    grad = np.ones_like(z.data)
    z.backward(Tensor(grad))
    tz.backward(tensor(grad))

    # 和老大哥 pytorch保持一致就行了
    assert np.allclose(x.grad.data, tx.grad.numpy())
    assert np.allclose(y.grad.data, ty.grad.numpy())
