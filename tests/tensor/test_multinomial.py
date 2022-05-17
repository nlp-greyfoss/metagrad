from metagrad.tensor import Tensor


def test_simple_multinomial():
    weights = Tensor([0, 10, 3, 0])
    x = Tensor.multinomial(weights, 5, replace=True)
    print(x)


def test_multinomial():
    weights = Tensor([[0, 10, 3, 0], [5, 0, 0, 5]])
    x = Tensor.multinomial(weights, 5, replace=True)
    print(x)