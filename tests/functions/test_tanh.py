import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor, debug_mode, cuda

device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")


def test_simple_tanh():
    x = 2.0

    mx = Tensor(x, requires_grad=True, device=device, dtype=np.float32)
    y = F.tanh(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tanh(tx)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_tanh():
    x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)

    with debug_mode():
        mx = Tensor(x, requires_grad=True, device=device)
        y = F.tanh(mx)

        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tanh(tx)
        assert np.allclose(y.data, ty.data)

        y.sum().backward()
        ty.sum().backward()
        assert np.allclose(mx.grad.data, tx.grad.data)
