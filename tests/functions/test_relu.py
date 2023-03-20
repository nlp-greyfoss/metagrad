import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor, debug_mode, cuda

device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
print(f"current device:{device}")


def test_simple_relu():
    x = 2.0

    mx = Tensor(x, requires_grad=True, device=device)
    y = F.relu(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.relu(tx)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad, tx.grad)


def test_relu():
    x = np.array([[-1, 0, 1], [-2, -3, 2]], np.float64)

    with debug_mode():
        mx = Tensor(x, requires_grad=True, device=device)
        y = F.relu(mx)

        tx = torch.tensor(x, requires_grad=True)
        ty = torch.relu(tx)

        assert np.allclose(y.data, ty.data)

        y.sum().backward()
        ty.sum().backward()

        assert np.allclose(mx.grad, tx.grad)
