import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_chunk():
    x = np.arange(12).astype(np.float32)

    mx = Tensor(x, requires_grad=True)

    tx = torch.tensor(x, requires_grad=True)

    my = F.chunk(mx, 2)
    ty = torch.chunk(tx, 2)

    # 这里返回的是列表
    assert isinstance(my, list)

    assert np.allclose(my[0].data, ty[0].data)
    assert np.allclose(my[-1].data, ty[-1].data)

    ty0 = ty[0] * 2
    ty1 = ty[1] * 3

    my0 = my[0] * 2
    my1 = my[1] * 3

    (my0 + my1).sum().backward()
    (ty0 + ty1).sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
