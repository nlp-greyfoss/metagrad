from metagrad.tensor import Tensor

def test_size_simple():
    x = Tensor([1, 2, 3, 4, 5, 6, 7])

    assert x.size() == 7


def test_size():
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)

    assert x.size() == 6

    assert x.size(0) == 2
    assert x.size(1) == 3
