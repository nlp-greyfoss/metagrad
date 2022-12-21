import numpy as np
import torch

from metagrad.tensor import Tensor
import metagrad.functions as F


def test_simple_leaky_relu():
    x = 2.0

    mx = Tensor(x, requires_grad=True)
    y = F.leaky_relu(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.nn.functional.leaky_relu(tx)

    assert np.allclose(y.data, ty.data)

    y.backward(np.array(2))
    ty.backward(torch.tensor(2))

    assert np.allclose(mx.grad, tx.grad)


def test_leaky_relu():
    x = np.array([[-1, 0, 1], [-2, -3, 2]], np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.leaky_relu(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.nn.functional.leaky_relu(tx)

    assert np.allclose(y.data, ty.data)

    y.backward(np.ones_like(mx))
    ty.backward(torch.ones_like(tx))

    assert np.allclose(mx.grad, tx.grad)
