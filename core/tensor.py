from typing import Union, Tuple

import numpy as np

# 默认数据类型
_type = np.float32


# 可以转换为Numpy数组的类型
Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    """
    :param arrayable:
    :return:
    """
    if isinstance(arrayable, np.ndarray):
        # 如果本身是ndarray
        return arrayable
    # 转换为Numpy数组
    return np.array(arrayable, dtype=_type)


Tensorable = Union["Tensor", float, np.ndarray]


def ensure_tensor(tensoralbe: Tensorable) -> "Tensor":
    '''
    确保是Tensor对象
    '''
    if isinstance(tensoralbe, Tensor):
        return tensoralbe
    return Tensor(tensoralbe)


class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False) -> None:
        '''
        初始化Tensor对象
        Args:
            data: 数据
            requires_grad: 是否需要计算梯度
        '''

        # data 是 np.ndarray
        self._data = ensure_array(data)

        self.requires_grad = requires_grad
        # 保存该Tensor的梯度
        self._grad = None

        if self.requires_grad:
            self.zero_grad()

        # 用于计算图的内部变量
        self._ctx = None

    @property
    def grad(self):
        return self._grad

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = ensure_array(new_data)
        # 重新赋值后就没有梯度了
        self._grad = None

    # ****一些常用属性****
    @property
    def shape(self) -> Tuple:
        '''返回Tensor各维度大小的元素'''
        return self.data.shape

    @property
    def ndim(self) -> int:
        '''返回Tensor的维度个数'''
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        '''返回Tensor中数据的类型'''
        return self.data.dtype

    @property
    def size(self) -> int:
        '''
        返回Tensor中元素的个数 等同于np.prod(a.shape)
        Returns:
        '''
        return self.data.size

    def zero_grad(self) -> None:
        '''
        将梯度初始化为0
        Returns:

        '''
        self._grad = Tensor(np.zeros_like(self.data, dtype=_type))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    def assign(self, x) -> "Tensor":
        '''将x的值赋予当前Tensor'''
        x = ensure_tensor(x)
        # 维度必须一致
        assert x.shape == self.shape
        self.data = x.data
        return self

    def numpy(self) -> np.ndarray:
        """转换为Numpy数组"""
        return self.data


if __name__ == '__main__':
    t = Tensor(range(10))
    print(t)
    print(t.shape)
    print(t.size)
    print(t.dtype)
