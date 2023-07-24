import copy
import math

import metagrad.module as nn
from metagrad import functions as F
from metagrad import Tensor


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    产生N个等同的层
    Args:
        module: 要克隆的module
        n: 克隆的次数

    Returns:

    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    p_attn = F.softmax(scores, -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn @ value, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d: int, dropout: float = 0.1) -> None:
        """
        多头注意力的初始化
        Args:
            h: 多头的个数
            d: 模型的维度d
            dropout: dropout比率
        """
        super().__init__()
        # 我们假设d_v == d_k
        self.d_k = d // h
        self.h = h
        # 分别表示Q,K,V和最后的线性投影
        self.linears = clones(nn.Linear(d, d), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> None:
        """

        Args:
            query: 形状  ()
            key:   形状  ()
            value: 形状  ()

        Returns:
        """

        num_batches = query.size(0)

        # ？
        query, key, value = [
            lin(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, dropout=self.dropout
        )

        x = (x.transpose(1, 2).view(num_batches, -1, self.h * self.d_k))

        return self.linears[-1](x)
