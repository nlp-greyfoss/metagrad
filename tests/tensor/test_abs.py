from metagrad.tensor import Tensor


def test_abs():
    x = Tensor([-1, 200, 0], requires_grad=True)
    y = abs(x)

    assert y.data.tolist() == [1, 200, 0]

    y.backward([1, 1, 1, ])

    assert x.grad.data.tolist() == [-1, 1, 0]
