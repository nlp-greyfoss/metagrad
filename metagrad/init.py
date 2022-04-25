import math
from typing import Tuple

from metagrad.tensor import Tensor, no_grad

'''
各种初始化方法，主要实现了He初始化和Xavier初始化。
为了简单，没有考虑gain，除非真的影响挺大。
'''


def _no_grad_uniform_(tensor: Tensor, a, b) -> Tensor:
    with no_grad():
        return tensor.uniform_(a, b)


def _no_grad_normal_(tensor, mean, std):
    with no_grad():
        return tensor.normal_(mean, std)


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple:
    dimensions = tensor.ndim

    assert dimensions == 2
    # 一般W的维度是[fan_out x fan_in]
    fan_in = tensor.size(1)  # 输入大小
    fan_out = tensor.size(0)  # 输出大小

    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> Tensor:
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    return _no_grad_normal_(tensor, mean, std)


def xavier_uniform_(tensor: Tensor) -> Tensor:
    '''
    Xavier初始化 均匀分布
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std  # 计算均匀分布的边界

    return _no_grad_uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor) -> Tensor:
    '''
      Xavier初始化 正态分布(高斯分布)
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0., std)


def kaiming_uniform_(tensor: Tensor, mode='fan_in') -> Tensor:
    '''
    kaiming初始化 均匀分布
    '''
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(6.0) * std
    return _no_grad_uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor, mode='fan_in') -> Tensor:
    '''
    kaiming初始化 正态分布(高斯分布)
    '''
    fan = _calculate_correct_fan(tensor, mode)
    std = math.sqrt(2.0 / fan)
    return _no_grad_normal_(tensor, 0., std)
