import torch

from metagrad.tensor import Tensor
import numpy as np


def test_simple_unsqueeze():
    x = np.array([1, 2, 3, 4])

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    my = mx.unsqueeze(0)
    ty = tx.unsqueeze(0)

    assert my.shape == ty.shape

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_squeeze():
    x = np.array([1, 2, 3, 4])

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    my = mx.unsqueeze(1)
    ty = tx.unsqueeze(1)

    assert my.shape == ty.shape

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
