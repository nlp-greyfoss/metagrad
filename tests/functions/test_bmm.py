import numpy as np
import torch

from metagrad.tensor import Tensor
import metagrad.functions as F


def test_bmm():
    x = np.random.randn(10, 3, 4)
    y = np.random.randn(10, 4, 5)

    mx = Tensor(x, requires_grad=True)
    my = Tensor(y, requires_grad=True)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)

    mz = F.bmm(mx, my)
    tz = torch.bmm(tx, ty)

    assert np.allclose(mz.data, tz.data)

    tz.sum().backward()
    mz.sum().backward()

    assert np.allclose(mx.grad, tx.grad)
    assert np.allclose(my.grad, ty.grad)
