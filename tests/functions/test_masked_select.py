import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_masked_select():
    x = np.random.randn(3, 4)
    print(x)

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    mask = x > 0.5

    print(mask)

    my = F.masked_select(mx, mask)
    ty = torch.masked_select(tx, torch.tensor(mask))

    assert np.array_equal(my.data, ty.data)

    ty.sum().backward()

    my.sum().backward()

    assert np.allclose(mx.grad, tx.grad)
