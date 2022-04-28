from collections import defaultdict

from tqdm import tqdm

import metagrad.module as nn
from metagrad.dataset import Dataset
from metagrad.tensor import Tensor

BOS_TOKEN = "<bos>"  # 句子开始标记
EOS_TOKEN = "<eos>"  # 句子结束标记
PAD_TOKEN = "<pad>"  # 填充标记
UNK_TOKEN = "<unk>"  # 未知词标记


def load_corpus():
    pass


class Vocabulary:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        # 如果传入了去重单词列表
        if tokens is not None:
            if UNK_TOKEN not in tokens:
                tokens = tokens + [UNK_TOKEN]
            # 构建id2word和word2id
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

            self.unk = self.token_to_idx[UNK_TOKEN]

    @classmethod
    def build(cls, text, min_freq=2, reserved_tokens=None):
        '''
        构建词表
        :param text: 处理好的(分词、去掉特殊符号等)text
        :param min_freq: 最小单词频率
        :param reserved_tokens: 保留的标记
        :return:
        '''
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        unique_tokens = [UNK_TOKEN] + (reserved_tokens if reserved_tokens else [])
        unique_tokens += [token for token, freq in token_freqs.items() \
                          if freq >= min_freq and token != UNK_TOKEN]
        return cls(unique_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        '''得到token对应的id'''
        return self.token_to_idx.get(token, self.unk)

    def to_ids(self, tokens):
        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


class CBOWDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            # 如果句子长度不足以构建(上下文,目标词)训练样本，则跳过
            if len(sentence) < window_size * 2 + 1:
                continue
            for i in range(window_size, len(sentence) - window_size):
                # 分别取i左右window_size个单词
                context = sentence[i - window_size:i] + sentence[i + 1:i + window_size]
                # 目标词：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(examples):
        '''
        自定义整理函数
        :param examples:
        :return:
        '''
        inputs = Tensor([ex[0] for ex in examples])
        targets = Tensor([ex[1] for ex in examples])
        return inputs, targets


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



# 先试简单的语料库
embedding_dim = 64
window_size = 2
hidden_dim = 128
batch_size = 5
num_epoch = 10


