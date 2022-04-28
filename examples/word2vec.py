
import metagrad.module as nn
from metagrad.tensor import Tensor


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        # 词向量层，即权重矩阵W
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出层，包含权重矩阵W'
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        # 得到所有上下文嵌入向量
        embeds = self.embeddings(inputs)
        # 计算均值，得到隐藏层向量，作为目标词的上下文表示
        hidden = embeds.mean(axis=1, keepdims=True)
        output = self.output(hidden)
        return output
