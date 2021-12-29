from typing import Union

from core.tensor import Tensor, Arrayable


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor]) -> None:
        # Parameter都是需要计算梯度的
        if isinstance(data, Tensor):
            data = data.data

        super().__init__(data, requires_grad=True)
