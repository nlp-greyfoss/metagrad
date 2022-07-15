import numpy as np
import torch

from metagrad.tensor import Tensor


def test_simple_argmin():
    x = np.random.randn(4, 4)
    mx = Tensor(x)
    tx = torch.tensor(x)

    assert mx.argmin().item() == tx.argmin().item()


def test_argmin():
    x = np.random.randn(4, 4)
    mx = Tensor(x)
    tx = torch.tensor(x)

    assert mx.argmin(axis=1).data.tolist() == tx.argmin(dim=1).data.tolist()
