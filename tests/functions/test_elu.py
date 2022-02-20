import numpy as np
import torch

from metagrad.tensor import Tensor
import metagrad.functions as F


def test_simple_elu():
    x = 2.0

    mx = Tensor(x, requires_grad=True)
    y = F.elu(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.nn.functional.elu(tx)

    assert np.allclose(y.data, ty.data)

    y.backward(2)
    ty.backward(torch.tensor(2))

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_elu():
    x = np.array([[-1, 0, 1], [-2, -3, 2]], np.float32)
    mx = Tensor(x, requires_grad=True)
    y = F.elu(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.nn.functional.elu(tx)

    assert np.allclose(y.data, ty.data)

    y.backward(Tensor.ones_like(mx))
    ty.backward(torch.ones_like(tx))

    print(mx.grad)
    print(tx.grad)
    assert np.allclose(mx.grad.data, tx.grad.data)
