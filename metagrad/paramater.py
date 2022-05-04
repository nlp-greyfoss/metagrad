from typing import Union

from metagrad.tensor import Tensor, Arrayable


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor], dtype=None, device=None) -> None:
        # Parameter都是需要计算梯度的
        super().__init__(data, requires_grad=True, dtype=dtype, device=device)
