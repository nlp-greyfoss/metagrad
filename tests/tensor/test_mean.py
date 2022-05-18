from metagrad.tensor import Tensor


def test_simple_mean():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x.mean()

    assert y.data == 2

    y.backward()
    # 均值的梯度均分到了每个元素上
    assert x.grad.data.tolist() == [1 / 3, 1 / 3, 1 / 3]


def test_mean_with_grad():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x.mean()

    y.backward(Tensor(3))

    assert x.grad.data.tolist() == [1, 1, 1]


def test_matrix_mean():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    y = x.mean()
    assert y.data == 3.5

    y.backward()

    assert x.grad.data.tolist() == [[1 / 6, 1 / 6, 1 / 6], [1 / 6, 1 / 6, 1 / 6]]


def test_matrix_with_axis():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    y = x.mean(0)  # keepdims = False

    assert y.shape == (3,)
    assert y.data.tolist() == [2.5, 3.5, 4.5]

    y.backward([1, 1, 1])

    assert x.grad.data.tolist() == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]


def test_matrix_with_keepdims():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    y = x.mean(axis=1, keepdims=True)  # keepdims = True
    assert y.shape == (2, 1)
    assert y.data.tolist() == [[2], [5]]
    y.backward([[3], [3]])

    assert x.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
