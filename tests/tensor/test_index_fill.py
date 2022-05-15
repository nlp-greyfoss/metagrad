import numpy as np

from metagrad.tensor import Tensor


def test_simple_index_fill():
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float)
    index = Tensor([0, 2])
    x.index_fill_(1, index, -1)
    print(x)
