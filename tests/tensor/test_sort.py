import numpy as np

from metagrad.tensor import Tensor


def test_simple_sort():
    data = np.array([2, 3, 4, 1])
    mx = Tensor(data, requires_grad=True)
    sorted_value, indices = mx.sort()

    assert sorted_value.tolist() == [1, 2, 3, 4]
    # sorted_value[0] == 1 对应原数组的索引3
    sorted_value[0].sum().backward()
    # 索引只有索引3的位置才有grad
    assert mx.grad.tolist() == [0, 0, 0, 1]


def test_sort_axis0():
    data = np.array([[2, 3, 4, 1], [1, 3, 2, 4]])
    mx = Tensor(data, requires_grad=True)
    sorted_value, indices = mx.sort(0)

    sorted_value[0].sum().backward()

    assert mx.grad.tolist() == [[0., 1., 0., 1.], [1., 0., 1., 0.]]


def test_sort_axis1():
    data = np.array([[2, 3, 4, 1], [1, 3, 2, 4]])
    mx = Tensor(data, requires_grad=True)
    sorted_value, indices = mx.sort(1)

    assert sorted_value.tolist() == [[1, 2, 3, 4], [1, 2, 3, 4]]

    sorted_value[0].sum().backward()

    assert mx.grad.tolist() == [[1, 1, 1, 1], [0, 0, 0, 0]]


def test_sort_descending():
    data = np.array([[2, 3, 4, 1], [1, 3, 2, 4]])
    mx = Tensor(data, requires_grad=True)
    # 在轴0上(按列)逆序排序
    sorted_value, indices = mx.sort(0, descending=True)

    assert sorted_value.tolist() == [[2., 3., 4., 4.],
                                     [1., 3., 2., 1.]]
    # 默认用不稳定的快排算法，3发生了交换
    assert indices.tolist() == [[0, 1, 0, 1],
                                [1, 0, 1, 0]]
    sorted_value[0].sum().backward()

    assert mx.grad.tolist() == [[1., 0., 1., 0.],
                                [0., 1., 0., 1.]]
