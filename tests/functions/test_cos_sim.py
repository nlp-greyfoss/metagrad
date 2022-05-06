import numpy as np

from metagrad import Tensor
import metagrad.functions as F
import torch


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def test_simple_cos_sim():
    # x = np.array([2, 1, 2, 3, 2, 9], dtype=np.float32)
    # y = np.array([3, 4, 2, 4, 5, 5], dtype=np.float32)

    x = np.random.randn(100, 128)
    y = np.random.randn(100, 128)

    mx = Tensor(x)
    my = Tensor(y)
    # mz = F.cos_sim(mx, my, axis=None)

    tx = torch.tensor(x)
    ty = torch.tensor(y)

    tz = torch.cosine_similarity(tx, ty, eps=0)
    print(tz)
    print(cos_sim(tx, ty))
    assert tz == cos_sim(tx, ty)
    #
    # print(mz)
    # print(tz)
    # assert np.allclose(mz.data, tz.data)
