import copy
import math

import numpy as np

import metagrad.module as nn
from metagrad import functions as F
from metagrad import Tensor
from metagrad.paramater import Parameter


# ─────────────────────────────────────────────
#  LayerNorm（手写）
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = Parameter(Tensor.ones(features))   # 缩放参数，初始化为 1
        self.beta  = Parameter(Tensor.zeros(features))  # 平移参数，初始化为 0
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., features]
        mean = x.mean(-1, keepdims=True)   # 最后一维均值
        std  = x.std(-1, keepdims=True)    # 最后一维标准差
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# ─────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None,
                                 dropout: nn.Dropout = None) -> Tensor:
    """
    缩放点积注意力
    Args:
        query: [batch_size, h, q_len, d_k]
        key:   [batch_size, h, k_len, d_k]
        value: [batch_size, h, k_len, d_v]
        mask:  可广播到 [batch_size, h, q_len, k_len]，1=关注 0=屏蔽
        dropout: Dropout 层
    Returns:
        [batch_size, h, q_len, d_v]
    """
    d_k = query.size(-1)
    # scores: [batch_size, h, q_len, k_len]
    scores = F.bmm(query, key.permute(0, 1, 3, 2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # weights: [batch_size, h, q_len, k_len]
    weights = F.softmax(scores, axis=-1)
    if dropout:
        weights = dropout(weights)
    # [batch_size, h, q_len, d_v]
    return F.bmm(weights, value)


def generate_mask(src: Tensor, pad: int = 0) -> Tensor:
    """
    生成 padding mask
    Args:
        src: [batch_size, seq_len]
        pad: <pad> token 的 id
    Returns:
        [batch_size, 1, 1, seq_len]，1=有效 0=填充
    """
    return (src != pad).unsqueeze(1).unsqueeze(2)


def generate_subsequent_mask(size: int) -> Tensor:
    """
    生成因果掩码（下三角矩阵），防止 Decoder 看到未来 token
    Returns:
        [1, 1, size, size]，1=可关注 0=屏蔽
    """
    mask = np.tril(np.ones((1, 1, size, size), dtype=np.float32))
    return Tensor(mask)


# ─────────────────────────────────────────────
#  位置编码
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构造固定位置编码矩阵 [max_len, d_model]
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        # 存为普通属性（不参与梯度更新），shape: [1, max_len, d_model]
        self.pe = Tensor(pe[np.newaxis, :, :])

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
#  多头注意力
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        # Q、K、V 各自独立的线性投影
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        # 输出线性层
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        batch_size = query.size(0)
        # 线性投影 + 拆分多头
        # [batch_size, seq_len, embed_dim] -> [batch_size, h, seq_len, d_k]
        query = self.q(query).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        key   = self.k(key).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)   # Fix: self.k
        value = self.v(value).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)  # Fix: self.v

        # attn_outputs: [batch_size, h, seq_len, d_k]
        attn_outputs = scaled_dot_product_attention(query, key, value, mask, self.dropout)
        # 合并多头: [batch_size, seq_len, h * d_k]
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).view(batch_size, -1, self.h * self.d_k)
        return self.linear(attn_outputs)


# ─────────────────────────────────────────────
#  前馈网络
# ─────────────────────────────────────────────

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────────
#  Encoder 层
# ─────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, norm_first: bool = False):
        """
        Args:
            norm_first: True → Pre-LN；False（默认）→ Post-LN
        """
        super().__init__()
        self.d_model = d_model
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        x = src
        if self.norm_first:
            # Pre-LN: 残差基是原始 x，norm 在子层内部
            x = x + self.dropout1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            # Post-LN: 子层计算后再 norm
            x = self.norm1(x + self.dropout1(self.attn(x, x, x, src_mask)))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


# ─────────────────────────────────────────────
#  Decoder 层
# ─────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, norm_first: bool = False):
        super().__init__()
        self.d_model = d_model
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        # 1. 带因果掩码的自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. 编码器-解码器交叉注意力（Q 来自 Decoder，K/V 来自 Encoder 输出）
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Tensor = None, memory_mask: Tensor = None) -> Tensor:
        """
        Args:
            tgt:         解码器输入 [batch_size, tgt_len, d_model]
            memory:      编码器输出 [batch_size, src_len, d_model]
            tgt_mask:    自注意力掩码（因果+padding）[batch_size, 1, tgt_len, tgt_len]
            memory_mask: 交叉注意力掩码（src padding） [batch_size, 1, 1, src_len]
        """
        x = tgt
        if self.norm_first:
            # Pre-LN
            x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
            x = x + self.dropout2(self.cross_attn(self.norm2(x), memory, memory, memory_mask))
            x = x + self.dropout3(self.feed_forward(self.norm3(x)))
        else:
            # Post-LN
            x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
            x = self.norm2(x + self.dropout2(self.cross_attn(x, memory, memory, memory_mask)))
            x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x


# ─────────────────────────────────────────────
#  Encoder / Decoder 容器（多层堆叠）
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = LayerNorm(encoder_layer.d_model)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = LayerNorm(decoder_layer.d_model)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Tensor = None, memory_mask: Tensor = None) -> Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
#  完整 Transformer 模型
# ─────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 pad_idx: int = 0, norm_first: bool = False):
        super().__init__()
        self.pad_idx = pad_idx

        # Embedding + 位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Encoder / Decoder
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, norm_first)
        decoder_layer = TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, norm_first)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出投影到目标词表
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src_emb = self.pos_encoding(self.src_embedding(src))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor,
               tgt_mask: Tensor, memory_mask: Tensor) -> Tensor:
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        return self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Args:
            src: [batch_size, src_len]  源序列（含 padding）
            tgt: [batch_size, tgt_len]  目标序列（训练时 teacher forcing）
        Returns:
            [batch_size, tgt_len, tgt_vocab_size]  未归一化的 logits
        """
        # Encoder padding mask: [batch_size, 1, 1, src_len]
        src_mask = generate_mask(src, self.pad_idx)

        # Decoder 掩码 = padding mask & 因果掩码
        # padding mask: [batch_size, 1, 1, tgt_len]
        # 因果掩码:      [1, 1, tgt_len, tgt_len]
        # 广播后取 min → [batch_size, 1, tgt_len, tgt_len]
        tgt_len = tgt.size(1)
        tgt_padding_mask = generate_mask(tgt, self.pad_idx)
        tgt_causal_mask = generate_subsequent_mask(tgt_len)
        tgt_mask = Tensor(np.minimum(tgt_padding_mask.data, tgt_causal_mask.data))

        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.output_projection(output)


# ─────────────────────────────────────────────
#  验证示例
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from tokenizer import BPETokenizer

    D_MODEL   = 64
    NUM_HEADS = 4
    D_FF      = 128

    # ── 1. TransformerEncoderLayer ───────────────────────────────
    print("=" * 55)
    print("1. TransformerEncoderLayer")

    src = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]))
    embedding = nn.Embedding(1000, D_MODEL)
    src_mask = generate_mask(src)
    enc_layer = TransformerEncoderLayer(D_MODEL, NUM_HEADS, dim_feedforward=D_FF)
    enc_out = enc_layer(embedding(src), src_mask)
    print(f"  src {src.shape} → enc_out {enc_out.shape}")   # (2, 5, 64)

    # ── 2. TransformerDecoderLayer ───────────────────────────────
    print("\n2. TransformerDecoderLayer")

    tgt = Tensor(np.array([[1, 2, 3, 4], [5, 6, 0, 0]]))
    tgt_embedding = nn.Embedding(800, D_MODEL)
    tgt_len = tgt.size(1)
    tgt_mask = Tensor(np.minimum(
        generate_mask(tgt).data,
        generate_subsequent_mask(tgt_len).data,
    ))
    dec_layer = TransformerDecoderLayer(D_MODEL, NUM_HEADS, dim_feedforward=D_FF)
    dec_out = dec_layer(tgt_embedding(tgt), enc_out, tgt_mask, src_mask)
    print(f"  tgt {tgt.shape}, memory {enc_out.shape} → dec_out {dec_out.shape}")  # (2, 4, 64)

    # ── 3. 完整 Transformer（硬编码 token id）────────────────────
    print("\n3. Full Transformer (hardcoded IDs)")

    src = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]))
    tgt = Tensor(np.array([[1, 2, 3, 4], [5, 6, 0, 0]]))

    model = Transformer(
        src_vocab_size=5000,
        tgt_vocab_size=4000,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=D_FF,
    )
    logits = model(src, tgt)
    print(f"  src {src.shape}, tgt {tgt.shape} → logits {logits.shape}")  # (2, 4, 4000)

    # ── 4. Tokenizer + Transformer 端到端 ────────────────────────
    print("\n4. Tokenizer + Transformer (end-to-end)")

    corpus = [
        "The cat sat on the mat.",
        "A dog ran in the park.",
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning is a subset of machine learning.",
        "Transformer models use self attention mechanisms.",
        "Natural language processing is fascinating.",
    ]
    tok = BPETokenizer(vocab_size=300)
    tok.train(corpus)

    pad_id = tok.vocab['<pad>']
    bos_id = tok.vocab['<bos>']
    vocab_size = len(tok.vocab)
    print(f"  vocab size: {vocab_size}")

    src_text = "The cat sat on the mat."
    tgt_text = "A dog ran in the park."
    src_ids = tok.encode(src_text)
    tgt_ids = [bos_id] + tok.encode(tgt_text)   # decoder 输入以 <bos> 开头
    print(f"  src ({len(src_ids)} tokens): {src_ids}")
    print(f"  tgt ({len(tgt_ids)} tokens): {tgt_ids}")

    def _pad(ids, length):
        return ids + [pad_id] * (length - len(ids))

    max_len = max(len(src_ids), len(tgt_ids))
    src = Tensor(np.array([_pad(src_ids, max_len)], dtype=np.int32))
    tgt = Tensor(np.array([_pad(tgt_ids, max_len)], dtype=np.int32))

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=D_FF,
        pad_idx=pad_id,
    )
    logits = model(src, tgt)
    print(f"  logits: {logits.shape}")   # (1, tgt_len, vocab_size)

    # 贪心解码（未训练，仅演示接口）
    pred_ids = np.argmax(logits.data, axis=-1)[0].tolist()
    print(f"  greedy decode: {tok.decode(pred_ids)!r}")
