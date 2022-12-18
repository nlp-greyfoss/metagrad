from metagrad.tensor import Tensor, cuda

device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")


def test_abs():
    x = Tensor([-1, 200, 0], requires_grad=True, device=device)
    y = abs(x)

    assert y.data.tolist() == [1, 200, 0]

    y.backward([2, 2, 2])

    assert x.grad.data.tolist() == [-2, 2, 0]
