import numpy as np
import torch

from metagrad.tensor import Tensor


def test_simple_argmax():
    x = np.random.randn(4, 4)
    mx = Tensor(x)
    tx = torch.tensor(x)

    assert mx.argmax().item() == tx.argmax().item()


def test_argmax():
    x = np.random.randn(4, 4)
    mx = Tensor(x)
    tx = torch.tensor(x)

    assert mx.argmax(axis=1).data.tolist() == tx.argmax(dim=1).data.tolist()
