import math
import os
import random
from collections import defaultdict
from typing import Tuple

from tqdm import tqdm

import metagrad.module as nn
from metagrad import Tensor, cuda
from metagrad import functions as F
from metagrad import init
from metagrad.dataloader import DataLoader
from metagrad.dataset import TensorDataset
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import Adam
from metagrad.tensor import no_grad
from metagrad.utils import grad_clipping

"""
数据集来自已经分好词的版本： https://github.com/multi30k/dataset/tree/master/data/task1/tok
"""

base_path = "../data/de-en"


def build_nmt_pair(src_path, tgt_path, reverse=False):
    """
    构建机器翻译source-target对
    :param src_path: 源语言目录
    :param tgt_path: 目标语言目录
    :param reverse:  是否逆序源语言
    :return: 分好词的source和target
    """
    source, target = [], []
    with open(src_path, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    with open(tgt_path, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()

    for src, tgt in zip(source_lines, target_lines):
        src_tokens = src.split()
        if reverse:
            src_tokens.reverse()
        tgt_tokens = tgt.split()

        source.append(src_tokens)
        target.append(tgt_tokens)

    return source, target


# source, target = build_nmt_pair(os.path.join(base_path, "val.de"), os.path.join(base_path, "val.en"))
# print(source[:2])
# print(target[:2])

class Vocabulary:
    BOS_TOKEN = "<bos>"  # 句子开始标记
    EOS_TOKEN = "<eos>"  # 句子结束标记
    PAD_TOKEN = "<pad>"  # 填充标记
    UNK_TOKEN = "<unk>"  # 未知词标记

    def __init__(self, tokens=None):
        self._idx_to_token = list()
        self._token_to_idx = dict()

        # 如果传入了去重单词列表
        if tokens is not None:
            if self.UNK_TOKEN not in tokens:
                tokens = tokens + [self.UNK_TOKEN]
            # 构建id2word和word2id
            for token in tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

            self.unk = self._token_to_idx[self.UNK_TOKEN]

    @classmethod
    def build(cls, text, min_freq=2, reserved_tokens=None):
        '''
        构建词表
        :param text: 处理好的(分词、去掉特殊符号等)text
        :param min_freq: 最小单词频率
        :param reserved_tokens: 预先保留的标记
        :return:
        '''
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        unique_tokens = (reserved_tokens if reserved_tokens else []) + [cls.UNK_TOKEN]
        unique_tokens += [token for token, freq in token_freqs.items() \
                          if freq >= min_freq and token != cls.UNK_TOKEN]
        return cls(unique_tokens)

    def __len__(self):
        return len(self._idx_to_token)

    def __getitem__(self, tokens):
        '''得到tokens对应的id'''
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    @property
    def id2token(self):
        '''返回idx_to_token列表'''
        return self._idx_to_token

    def token(self, indices):
        '''根据索引获取token'''
        if not isinstance(indices, (list, tuple)):
            return self._idx_to_token[indices]

        return [self._idx_to_token[index] for index in indices]

    def to_tokens(self, indices):
        return self.token(indices)

    def save(self, path):
        with open(path, 'w') as f:
            f.write("\n".join(self.id2token))

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            tokens = f.read().split('\n')
        return cls(tokens)


# min_freq = 2
# source, target = build_nmt_pair(os.path.join(base_path, "train.de"), os.path.join(base_path, "train.en"))
#
# reserved_tokens = [Vocabulary.PAD_TOKEN, Vocabulary.BOS_TOKEN, Vocabulary.EOS_TOKEN]
# src_vocab = Vocabulary.build(source, min_freq=min_freq, reserved_tokens=reserved_tokens)
# tgt_vocab = Vocabulary.build(target, min_freq=min_freq, reserved_tokens=reserved_tokens)


# print(len(src_vocab), len(tgt_vocab))


def truncate_pad(line, max_len, padding_token):
    """截断或填充文本序列"""
    if len(line) > max_len:
        return line[:max_len]  # 截断
    return line + [padding_token] * (max_len - len(line))  # 填充


# print(truncate_pad(src_vocab[source[0]], 20, src_vocab[Vocabulary.PAD_TOKEN]))


def build_array_nmt(lines, vocab, max_len=None):
    """将机器翻译的文本序列转换成小批量"""
    if not max_len:
        max_len = max(len(x) for x in lines)

    # 先转换成token对应的索引列表
    lines = [vocab[l] for l in lines]
    # 增加BOS和EOS token的索引
    lines = [[vocab[Vocabulary.BOS_TOKEN]] + l + [vocab[Vocabulary.EOS_TOKEN]] for l in lines]
    # max_len 应该加2了:额外的BOS和EOS ，并转换为seq_len, batch_size的形式
    array = Tensor([truncate_pad(l, max_len + 2, vocab[Vocabulary.PAD_TOKEN]) for l in lines])

    return array


# src_array = build_array_nmt(source, src_vocab, 20)
# print(src_array[:5])


def load_dataset_nmt(data_path=base_path, data_type="train", batch_size=32, min_freq=2, src_vocab=None, tgt_vocab=None,
                     shuffle=False):
    """
    加载机器翻译数据集
    :param data_path: 保存数据集的目录
    :param data_type: 数据集类型 train|test|val
    :param batch_size: 批大小
    :param min_freq: 最小单词次数
    :param src_vocab: 源词典
    :param tgt_vocab: 目标词典
    :param shuffle: 是否打乱
    :return:
    """

    source, target = build_nmt_pair(os.path.join(data_path, f"{data_type}.de"),
                                    os.path.join(data_path, f"{data_type}.en"),
                                    reverse=True)
    # 构建源和目标词表
    reserved_tokens = [Vocabulary.PAD_TOKEN, Vocabulary.BOS_TOKEN, Vocabulary.EOS_TOKEN]

    if src_vocab is None:
        src_vocab = Vocabulary.build(source, min_freq=min_freq, reserved_tokens=reserved_tokens)
    if tgt_vocab is None:
        tgt_vocab = Vocabulary.build(target, min_freq=min_freq, reserved_tokens=reserved_tokens)

    print(f'Source vocabulary size: {len(src_vocab)}, Target vocabulary size: {len(tgt_vocab)}')
    # 转换成批数据
    max_src_len = max([len(line) for line in source])
    max_tgt_len = max([len(line) for line in target])

    print(f"max_src_len: {max_src_len}, max_tgt_len:{max_tgt_len}")

    src_array = build_array_nmt(source, src_vocab, max_src_len)
    tgt_array = build_array_nmt(target, tgt_vocab, max_tgt_len)

    # 构建数据集
    dataset = TensorDataset(src_array, tgt_array)
    # 数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # 返回加载器和两个词表
    return data_loader, src_vocab, tgt_vocab


# train_dataset, src_vocab, tgt_vocab = load_dataset_nmt()
# for X, Y in train_dataset:
#    print('X:', X.shape)
#    print('Y:', Y.shape)
#    break


class Attention(nn.Module):
    def __init__(self, enc_hid_dim=None, dec_hid_dim=None, method: str = "dot") -> None:
        """

        Args:
            enc_hid_dim: 编码器的隐藏层大小
            dec_hid_dim: 解码器的隐藏层大小
            method:  dot | scaled_dot | general | bahdanau | concat
        Returns:

        """
        super().__init__()
        self.method = method
        self.encoder_hidden_size = enc_hid_dim
        self.decoder_hidden_size = dec_hid_dim

        if self.method not in ["dot", "scaled_dot", "general", "bahdanau", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if self.method not in ["dot", "scaled_dot"] and (
                self.encoder_hidden_size is None or self.decoder_hidden_size is None):
            raise ValueError(self.method, "must pass encoder and decoer hidden size parameter.")

        if self.method == "general":
            self.linear = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
        elif self.method == "bahdanau":
            self.W = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.U = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.v = nn.Linear(self.decoder_hidden_size, 1, bias=False)
        elif self.method == "concat":
            # concat
            self.linear = nn.Linear((self.encoder_hidden_size + self.decoder_hidden_size), self.decoder_hidden_size,
                                    bias=False)
            self.v = nn.Linear(self.decoder_hidden_size, 1, bias=False)

    def _score(self, hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """

        Args:
            hidden: (batch_size, decoder_hidden_size)  解码器前一时刻的隐藏状态
            encoder_outputs: (src_len, batch_size, encoder_hidden_size) 编码器的输出(隐藏状态)序列

        Returns:

        """

        src_len, batch_size, encoder_hidden_size = encoder_outputs.shape

        if self.method == "dot":
            # 这里假设编码器和解码器的隐藏层大小一致，如果不一致，不能直接使用点积注意力
            # (batch_size, hidden_size) * (src_len, batch_size, hidden_size) -> (src_len, batch_size, hidden_size)
            # (src_len, batch_size, hidden_size).sum(axis=2) -> (src_len, batch_size)
            return (hidden * encoder_outputs).sum(axis=2)
        elif self.method == "scaled_dot":
            # 和点积注意力类似，不过除以sqrt(batch_size)，也可以指定一个自定义的值
            return (hidden * encoder_outputs / math.sqrt(encoder_hidden_size)).sum(axis=2)
        elif self.method == "general":
            # energy = (src_len, batch_size, decoder_hidden_size)
            energy = self.linear(encoder_outputs)
            # (batch_size, decoder_hidden_size)  * (src_len, batch_size, decoder_hidden_size)
            # -> (src_len, batch_size, decoder_hidden_size)
            # .sum(axis=2) -> (src_len, batch_size)
            return (hidden * energy).sum(axis=2)
        elif self.method == "bahdanau":
            # hidden = (batch_size, decoder_hidden_size)
            # encoder_outputs = (src_len, batch_size, encoder_hidden_size)
            # W(hidden) -> (batch_size, decoder_hidden_size)
            # U(encoder_outputs) -> (src_len, batch_size, decoder_hidden_size)
            # energy = (src_len, batch_size, decoder_hidden_size)
            energy = F.tanh(self.W(hidden) + self.U(encoder_outputs))
            # v(energy) ->  (src_len, batch_size, 1)
            # squeeze -> (src_len, batch_size)
            return self.v(energy).squeeze(2)
        else:
            # concat
            # unsqueeze -> (1, batch_size, decoder_hidden_size)
            # hidden = (src_len, batch_size, decoder_hidden_size)
            hidden = hidden.unsqueeze(0).repeat(src_len, 1, 1)
            #  encoder_outputs = (src_len, batch_size, encoder_hidden_size)
            # cat -> (src_len, batch_size,  encoder_hidden_size +decoder_hidden_size)
            # linear -> (src_len, batch_size, decoder_hidden_size)
            energy = F.tanh(self.linear(F.cat((hidden, encoder_outputs), axis=2)))
            # v -> (src_len, batch_size , 1)
            # squeeze -> (src_len, batch_size)
            return self.v(energy).squeeze(2)

    def forward(self, hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        """

        Args:
            hidden: (batch_size, decoder_hidden_size)  解码器前一时刻的隐藏状态
            encoder_outputs: (src_len, batch_size, encoder_hidden_size) 编码器的输出(隐藏状态)序列

        Returns: 注意力权重， 上下文向量

        """
        # (src_len, batch_size)
        attn_scores = self._score(hidden, encoder_outputs)
        # (batch_size, src_len)
        attn_scores = attn_scores.T
        # (batch_size, 1, src_len)
        attention_weight = F.softmax(attn_scores, axis=1).unsqueeze(1)
        # encoder_outputs = (batch_size, src_len, num_hiddens)
        encoder_outputs = encoder_outputs.transpose((1, 0, 2))
        # context = (batch_size, 1, num_hiddens)
        context = F.bmm(attention_weight, encoder_outputs)
        # context = (1, batch_size, num_hiddens)
        context = context.transpose((1, 0, 2))

        return attention_weight, context


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int, num_layers: int, dropout: float) -> None:
        """
        基于GRU实现的编码器
        :param vocab_size: 源词表大小
        :param embed_size: 词嵌入大小
        :param num_hiddens: 隐藏层大小
        :param num_layers: GRU层数
        :param dropout:  dropout比率
        """

        super().__init__()

        # 嵌入层 获取输入序列中每个单词的嵌入向量 padding_idx不需要更新嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 这里固定为双向GRU实现 注意，这里默认batch_first为False
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout,
                          bidirectional=True)
        self.fc = nn.Linear(num_hiddens * 2, num_hiddens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码器的前向算法
        :param input_seq:  形状 (seq_len, batch_size)
        :return:
        """

        # (seq_len, batch_size, embed_size)
        embedded = self.dropout(self.embedding(input_seq))
        # embedded = self.embedding(input_seq)
        # outputs (seq_len, batch_size, 2 * num_hiddens)
        # hidden  (num_direction * num_layers, batch_size, num_hiddens)
        # hidden 堆叠了 [forward_1, backward_1, forward_2, backward_2, ...]
        # hidden[-2, :, :] 是最后一个前向 RNN
        # hidden[-1, :, :] 是最后一个反向 RNN
        # outputs 总是来自最后一层
        outputs, hidden = self.rnn(embedded)

        # 编码器初始隐藏状态是解码器最后一个前向和反向 RNN
        # 编码器的RNN喂给一个线性层来融合
        hidden = F.tanh(self.fc(F.cat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)))
        # outputs = (seq_len, batch_size,  num_hiddens)
        outputs = outputs[:, :, :num_hiddens] + outputs[:, :, :num_hiddens]

        # outputs (seq_len, batch_size,  num_hiddens)
        # hidden  (batch_size, num_hiddens)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int, num_layers: int, dropout: float,
                 attention: Attention) -> None:
        """
        基于GRU实现的解码器
        :param vocab_size: 目标词表大小
        :param embed_size:  词嵌入大小
        :param num_hiddens: 隐藏层大小
        :param num_layers:  层数
        :param dropout:  dropout比率
        :param attention:  Attention实例
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        #
        self.rnn = nn.GRU(input_size=embed_size + num_hiddens, hidden_size=num_hiddens, num_layers=num_layers,
                          dropout=dropout)
        # 将隐状态转换为词典大小维度
        self.fc_out = nn.Linear(num_hiddens, vocab_size)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input_seq: Tensor, hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        解码器的前向算法
        :param input_seq: 初始输入，这里为<bos> 形状 (batch_size, )
        :param hidden:  解码器的隐藏状态 (batch_size, num_hiddens)
        :param encoder_outputs: 编码器的输出的隐藏状态序列 形状 (src_len, batch_size, num_hiddens)
        :return:
        """
        # input = (1, batch_size)
        input_seq = input_seq.unsqueeze(0)
        # embedded = (1, batch_size, embed_size)
        embedded = self.dropout(self.embedding(input_seq))

        # attention = (batch_size, 1, src_len)
        # context = (1, batch_size, num_hiddens)
        attention, context = self.attention(hidden, encoder_outputs)

        # embedded_context = (1, batch_size, embed_size + hidden_dim)
        embedded_context = F.cat((embedded, context), 2)

        # output (1, batch_size, num_hiddens)
        # hidden  (1, batch_size, num_hiddens)
        output, hidden = self.rnn(embedded_context, hidden.unsqueeze(0))
        # prediction (batch_size, vocab_size)
        prediction = self.fc_out(output.squeeze(0))
        # hidden = (batch_size, num_hiddens)
        return prediction, hidden.squeeze(0)


class Seq2seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        """
        初始化seq2seq模型
        :param encoder: 编码器
        :param decoder: 解码器
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq: Tensor, target_seq: Tensor, teacher_forcing_ratio: float = 0.0) -> Tensor:
        """
        seq2seq的前向算法
        :param input_seq:  输入序列 (seq_len, batch_size)
        :param target_seq: 目标序列  (seq_len, batch_size)
        :param teacher_forcing_ratio:  强制教学比率
        :return:
        """

        tgt_len, batch_size = target_seq.shape
        # 保存了所有时间步的输出
        outputs = []
        # 这里我们只关心编码器输出的hidden
        # hidden  (num_layers, batch_size, num_hiddens)
        encoder_outputs, hidden = self.encoder(input_seq)

        # decoder_input (batch_size) 取BOS token
        decoder_input = target_seq[0, :]  # BOS_TOKEN
        # 这里从1开始，确保tgt[t]是下一个token
        for t in range(1, tgt_len):
            # output (batch_size, target_vocab_size)
            # hidden (num_layers, batch_size, num_hiddens)
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            # 保存到outputs
            outputs.append(output)
            # 随机判断是否强制教学
            teacher_force = random.random() < teacher_forcing_ratio
            # 如果teacher_force==True， 则用真实输入当成下一步的输入，否则用模型生成的
            # output.argmax(1) 在目标词表中选择得分最大的一个 (batch_size, 1)
            decoder_input = target_seq[t] if teacher_force else output.argmax(1)

        # 把outputs转换成一个Tensor 形状为： (tgt_len - 1, batch_size, target_vocab_size)
        return F.stack(outputs)


# Train

def init_weights(model):
    for name, param in model.named_parameters():
        init.uniform_(param, -0.08, 0.08)


# 参数定义
embed_size = 256
num_hiddens = 512
num_layers = 1
dropout = 0.5

batch_size = 16
max_len = 40

lr = 0.001
num_epochs = 10
min_freq = 2
clip = 1.0

tf_ratio = 0.5  # teacher force ratio

print_every = 1

device = cuda.get_device("cuda" if cuda.is_available() else "cpu")

# 加载训练集
train_iter, src_vocab, tgt_vocab = load_dataset_nmt(data_path=base_path, data_type="train", batch_size=batch_size,
                                                    min_freq=min_freq, shuffle=True)

# 加载验证集
valid_iter, src_vocab, tgt_vocab = load_dataset_nmt(data_path=base_path, data_type="val", batch_size=batch_size,
                                                    min_freq=min_freq, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

# 加载测试集
test_iter, src_vocab, tgt_vocab = load_dataset_nmt(data_path=base_path, data_type="test", batch_size=batch_size,
                                                   min_freq=min_freq, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

# 构建Attention
attention = Attention()
# 构建编码器
encoder = Encoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# 构建解码器
decoder = Decoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout, attention)

model = Seq2seq(encoder, decoder)
model.apply(init_weights)
model.to(device)

print(model)

TGT_PAD_IDX = tgt_vocab[Vocabulary.PAD_TOKEN]

print(f"TGT_PAD_IDX is {TGT_PAD_IDX}")

optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss(ignore_index=TGT_PAD_IDX)


def train_epoch(model, data_iter, optimizer, criterion, clip, device, tf_ratio):
    model.train()
    epoch_loss = 0

    for batch in data_iter:
        optimizer.zero_grad()
        inputs, targets = [x.to(device).T for x in batch]
        outputs = model(inputs, targets, tf_ratio)

        outputs = outputs.view(-1, outputs.shape[2])
        targets = targets[1:].view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        with no_grad():
            # 梯度裁剪
            grad_clipping(model, clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss


def evaluate(model, data_iter, criterion, device):
    model.eval()
    epoch_loss = 0
    with no_grad():
        for batch in data_iter:
            inputs, targets = [x.to(device).T for x in batch]
            outputs = model(inputs, targets, 0)  # 评估时不用teacher forcing

            outputs = outputs.view(-1, outputs.shape[2])
            targets = targets[1:].view(-1)

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

    return epoch_loss


def train(model, num_epochs, train_iter, valid_iter, optimizer, criterion, clip, device, tf_ratio):
    best_valid_loss = float('inf')
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training", leave=False):
        train_loss = train_epoch(model, train_iter, optimizer, criterion, clip, device, tf_ratio)
        valid_loss = evaluate(model, valid_iter, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save()
        tqdm.write(
            f"epoch {epoch:3d} , train loss: {train_loss:.4f} , validate loss: {valid_loss:.4f}, best validate loss: {best_valid_loss:.4f}")


train(model, num_epochs, train_iter, valid_iter, optimizer, criterion, clip, device, tf_ratio)
test_loss = evaluate(model, test_iter, criterion, device)
print(f"Test loss: {test_loss:.4f}")
