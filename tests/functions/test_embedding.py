import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_embedding():
    # 4 x 3
    weight = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

    x = 2

    m_weight = Tensor(weight, requires_grad=True)
    mx = Tensor(x)
    mx.data = np.array(2,dtype=int)
    my = F.embedding(mx, m_weight)

    t_weight = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    tx = torch.tensor(x)
    ty = torch.embedding(t_weight, tx)

    assert np.allclose(my.data, ty.data)
    #my.sum().backward()
    #ty.sum().backward()

    # assert np.allclose(mx.grad.data, tx.grad.data)

