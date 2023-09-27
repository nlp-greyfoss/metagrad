import copy
import math

import metagrad.module as nn
from metagrad import functions as F
from metagrad import Tensor


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None,
                                 dropout: nn.Dropout = None) -> Tensor:
    """
    缩放点积注意力实现函数
    Args:
        query: [batch_size, h, input_len, d_k]
        key:   [batch_size, h, input_len, d_k]
        value: [batch_size, h ,input_len, d_k]
        mask:  [batch_size, 1, 1, input_len]
        dropout: Dropout层

    Returns:

    """
    d_k = query.size(-1)
    # query [batch_size, h, input_len, d_k]
    # key.permute -> [batch_size, h, d_k, input_len]
    # 固定batch_size, self.h  -> (input_len, self.d_k)  x (self.d_k, input_len) = (input_len, input_len)
    #   -> [batch_size, self.h, input_len, input_len]
    # scores [batch_size, h, input_len, input_len]
    scores = F.bmm(query, key.permute(0, 1, 3, 2)) / math.sqrt(d_k)
    # 对于源序列来说，由于批次内语句长短不一，对于短的语句，需要填充<pad> token
    if mask is not None:
        # 根据mask，把填充的位置填-1e9，然后计算softmax的时候，-1e9的位置就被计算为0
        scores = scores.masked_fill(mask == 0, -1e9)
    # weights [batch_size, h, input_len, input_len]
    weights = F.softmax(scores, axis=-1)
    if dropout:
        weights = dropout(weights)
    # [batch_size, h, input_len, d_k]
    return F.bmm(weights, value)


def generate_mask(src: Tensor, pad: int = 0):
    """
    生成mask
    Args:
        src:  [batch_size, input_len]
        pad: 填充<pad>的id

    Returns:

    """
    # src_mask [batch_size, 1, 1, input_len]
    src_mask = (src != pad).unsqueeze(1).unsqueeze(2)
    return src_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_k = embed_dim // num_heads  # 计算每个头的维度
        self.h = num_heads
        # 定义Q,K,V映射
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        # 输出的那个线性变换
        self.linear = nn.Linear(embed_dim, embed_dim)
        # Dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        注意力的前向算法
        Args:
            query: 来自Encoder的嵌入向量 [batch_size, input_len, d_k]
            key:   来自Encoder的嵌入向量 [batch_size, input_len, d_k]
            value: 来自Encoder的嵌入向量 [batch_size, input_len, d_k]
            mask: 来自Encoder输入的mask [batch_size, 1, 1, input_len]

        Returns:

        """
        batch_size = query.size(0)
        # ====拆分head====
        # 线性映射后转换为形状 [batch_size, input_len, self.h, self.d_k]
        # 即h个d_k维度的query,key,value    embed_dim == h x d_k
        # permute -> [batch_size, self.h, input_len, self.d_k]
        query = self.q(query).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)  # transpose(1, 2)
        key = self.q(key).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        value = self.q(value).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)

        # attn_outputs [batch_size, h, input_len, d_k]
        attn_outputs = scaled_dot_product_attention(query, key, value, mask, self.dropout)
        # ====合并head====
        # 在计算时并没有将多个头分开计算，而是放在矩阵中一起运算
        # 这里可以直接通过view来执行类似拼接的操作，然后应用到最后一个线性层
        # permute -> [batch_size, input_len, h, d_k]
        # view -> [batch_size, input_len, h * d_k]
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).view(batch_size, -1, self.h * self.d_k)
        return self.linear(attn_outputs)


class PositionWiseFeedForward(nn.Module):
    '''
    实现FFN网路
    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        """

        Args:
            d_model: 模型大小
            d_ff: FF层的大小,2
            dropout:
        """
        super().__init__()
        # 将输入转换为d_ff维度
        self.linear1 = nn.Linear(d_model, d_ff)
        # 将d_ff转换回d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 经过三次变换： 线性->非线性->线性
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 norm_first: bool = False):
        """

        Args:
            d_model: 输入的特征个数
            num_heads: 多头个数
            dim_feedforward: FFN中的扩张的维度大小，通常会比d_model要大
            dropout:
            norm_first: 为True记为Pre-LN；默认为False对应的Post-LN。
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        x = src
        if self.norm_first:
            # 层归一化
            x = self.norm1(x)
            # 多头注意力 -> 残差连接
            x = x + self.dropout1(self.attn(x, x, x, src_mask))
            # 层归一化 -> FFN -> 残差连接
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            # 多头注意力 -> 残差连接 -> 层归一化
            x = self.norm1(x + self.dropout1(self.attn(x, x, x, src_mask)))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))

        return x


import numpy as np

embed_dim = 512
vocab_size = 5000

num_heads = 8
# 第二个样本包含两个填充
input_data = Tensor(
    np.array([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
)
# batch_size = 2
# seq_len = 5
batch_size, seq_len = input_data.shape

embedding = nn.Embedding(vocab_size, embed_dim)
# 模拟嵌入层
input_embeds = embedding(input_data)

mask = generate_mask(input_data)

# attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
#
# ffn = PositionWiseFeedForward(embed_dim, d_ff=2048)
#
# # 编码器中，query、key、value来自同一个嵌入
# values = attn(input_embeds, input_embeds, input_embeds, mask)
# print(values.shape)
# output = ffn(values)
# print(values.shape)

encoder_layer = TransformerEncoderLayer(embed_dim, num_heads,dim_feedforward=2048)
output = encoder_layer(input_embeds, mask)
print(output.shape)