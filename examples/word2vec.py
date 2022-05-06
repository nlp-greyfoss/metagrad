from collections import defaultdict

import numpy as np
from torch.nn.init import uniform_
from tqdm import tqdm

import metagrad.module as nn
from metagrad import cuda
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import SGD
from metagrad.tensor import Tensor
import metagrad.functions as F

BOS_TOKEN = "<bos>"  # 句子开始标记
EOS_TOKEN = "<eos>"  # 句子结束标记
PAD_TOKEN = "<pad>"  # 填充标记
UNK_TOKEN = "<unk>"  # 未知词标记

WEIGHT_INIT_RANGE = 0.1


class Vocabulary:
    def __init__(self, tokens=None):
        self._idx_to_token = list()
        self._token_to_idx = dict()

        # 如果传入了去重单词列表
        if tokens is not None:
            if UNK_TOKEN not in tokens:
                tokens = tokens + [UNK_TOKEN]
            # 构建id2word和word2id
            for token in tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

            self.unk = self._token_to_idx[UNK_TOKEN]

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

        unique_tokens = [UNK_TOKEN] + (reserved_tokens if reserved_tokens else [])
        unique_tokens += [token for token, freq in token_freqs.items() \
                          if freq >= min_freq and token != UNK_TOKEN]
        return cls(unique_tokens)

    def __len__(self):
        return len(self._idx_to_token)

    def __getitem__(self, token):
        '''得到token对应的id'''
        return self._token_to_idx.get(token, self.unk)

    def token(self, idx):
        assert 0 <= idx < len(self._idx_to_token), f"actual index : {idx} not between 0 and {len(self._idx_to_token)}"
        '''根据索引获取token'''
        return self._idx_to_token[idx]

    def to_ids(self, tokens):
        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        return [self._idx_to_token[index] for index in indices]


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
                context = sentence[i - window_size:i] + sentence[i + 1:i + window_size + 1]
                # 目标词：当前词
                target = sentence[i]
                self.data.append((context, target))

        self.data = np.asarray(self.data)

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


class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]

            for i in range(1, len(sentence) - 1):
                # 模型输入：当前词
                w = sentence[i]
                # 模型输出： 窗口大小内的上下文
                # max 和 min 防止越界取到非预期的单词
                left_context_index = max(0, i - window_size)
                right_context_index = min(len(sentence), i + window_size)
                context = sentence[left_context_index:i] + sentence[i + 1:right_context_index + 1]
                self.data.extend([(w, c) for c in context])

        self.data = np.asarray(self.data)

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
        uniform_(self.embeddings.weight, -WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE)
        # 输出层，包含权重矩阵W'
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        # 得到所有上下文嵌入向量
        embeds = self.embeddings(inputs)
        # 计算均值，得到隐藏层向量，作为目标词的上下文表示
        hidden = embeds.mean(axis=1)
        output = self.output(hidden)
        return output


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        # 得到输入词向量
        embeds = self.embeddings(inputs)
        # 根据输入词向量，对上下文进行预测
        output = self.output(embeds)
        return output


def load_corpus(corpus_path, min_freq=2):
    '''
    从corpus_path中读取预料
    :param corpus_path: 处理好的文本路径
    :return:
    '''
    with open(corpus_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    # 去掉空行，将文本转换为单词列表
    text = [[word for word in sentence.split(' ')] for sentence in lines if len(sentence) != 0]
    # 构建词典
    vocab = Vocabulary.build(text, min_freq=min_freq, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    print(f'vocab size:{len(vocab)}')
    # 构建语料:将单词转换为ID
    corpus = [vocab.to_ids(sentence) for sentence in text]

    return corpus, vocab


def train_cbow():
    embedding_dim = 64
    window_size = 3
    batch_size = 1024
    num_epoch = 10
    min_freq = 3  # 保留单词最少出现的次数

    corpus, vocab = load_corpus('data/xiyouji.txt', min_freq)
    # 构建数据集
    dataset = CBOWDataset(corpus, vocab, window_size=window_size)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

    print(f'current device:{device}')

    loss_func = CrossEntropyLoss()
    # 构建模型
    model = CBOWModel(len(vocab), embedding_dim)
    model.to(device)

    optimizer = SGD(model.parameters(), lr=1)
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f'Loss: {total_loss.item():.2f}')

    model.save()


def train_sg():
    embedding_dim = 64
    window_size = 3
    batch_size = 1024
    num_epoch = 10
    min_freq = 3  # 保留单词最少出现的次数

    # 读取文本数据，构建Skip-gram模型训练数据集
    corpus, vocab = load_corpus('data/xiyouji.txt', min_freq)
    dataset = SkipGramDataset(corpus, vocab, window_size=window_size)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    loss_func = CrossEntropyLoss()
    # 构建Skip-gram模型，并加载至device
    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
    model = SkipGramModel(len(vocab), embedding_dim)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=1)

    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Loss: {total_loss.item():.2f}")


if __name__ == '__main__':
    min_freq = 5  # 保留单词最少出现的次数
    embedding_dim = 64

    corpus, vocab = load_corpus('data/xiyouji.txt', min_freq)

    model = SkipGramModel(len(vocab), embedding_dim)
    model = model.load()
    weight = model.embeddings.weight.data
    import cupy

    s = cupy.sqrt((weight * weight).sum(1))
    weight /= s.reshape((s.shape[0], 1))  # normalize

    key = '大闹'
    idx = vocab[key]

    embed = weight[idx]
    embed = embed.reshape((-1, 1))
    print(embed.shape)

    print(weight.shape)

    score = weight @ embed
    score = score.squeeze()
    print(f'score shape:{len(score)}')

    count = 0
    for i in (-score).argsort():
        i = i.item()
        if vocab.token(i) == key:
            continue
        print('{0}: {1}'.format(vocab.token(i), score[i]))
        count += 1
        if count == 5:
            break
