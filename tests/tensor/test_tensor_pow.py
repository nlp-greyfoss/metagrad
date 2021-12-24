from core.tensor import Tensor


def test_simple_pow():
    x = Tensor(2, requires_grad=True)
    y = 2
    z = x ** y

    assert z.data == 4

    assert x.grad.data == 4
