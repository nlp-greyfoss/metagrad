import numpy as np
import torch

from metagrad.tensor import Tensor, debug_mode
import metagrad.functions as F


def test_simple_relu():
    x = 2.0

    mx = Tensor(x, requires_grad=True)
    y = F.relu(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.relu(tx)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_relu():
    x = np.array([[-1, 0, 1], [-2, -3, 2]], np.float32)

    with debug_mode():
        mx = Tensor(x, requires_grad=True)
        y = F.relu(mx)

        tx = torch.tensor(x, requires_grad=True)
        ty = torch.relu(tx)

        assert np.allclose(y.data, ty.data)

        y.sum().backward()
        ty.sum().backward()

        assert np.allclose(mx.grad.data, tx.grad.data)
