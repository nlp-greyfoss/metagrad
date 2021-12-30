from typing import Union

from core.tensor import Tensor, Arrayable


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor]) -> None:
        if isinstance(data, Tensor):
            data = data.data

        # Parameter都是需要计算梯度的
        super().__init__(data, requires_grad=True)
