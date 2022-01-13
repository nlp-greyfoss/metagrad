import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_cross_entropy():
    x = np.array([[0, 1, 2, 3], [4, 0, 2, 1]], np.float32)
    t = np.array([3, 0]).astype(np.int32)

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(t)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    mo = F.cross_entropy(mx, mt)
    to = torch.nn.functional.cross_entropy(tx, tt)
    assert mo.item() == to.item()

    mo.backward()
    to.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
