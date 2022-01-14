import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_sigmoid():
    x = 2.0

    mx = Tensor(x, requires_grad=True)
    y = F.sigmoid(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.sigmoid(tx)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_sigmoid():
    x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.sigmoid(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.sigmoid(tx)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
