from collections import defaultdict
import numpy as np

from metagrad import Tensor

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

    @property
    def id2token(self):
        '''返回idx_to_token列表'''
        return self._idx_to_token

    def token(self, idx):
        assert 0 <= idx < len(self._idx_to_token), f"actual index : {idx} not between 0 and {len(self._idx_to_token)}"
        '''根据索引获取token'''
        return self._idx_to_token[idx]

    def to_ids(self, tokens):
        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        return [self._idx_to_token[index] for index in indices]


def load_corpus(corpus_path, min_freq=2):
    '''
    从corpus_path中读取预料
    :param corpus_path: 处理好的文本路径
    :return:
    '''
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去掉空行，将文本转换为单词列表
    # 去掉换行符
    text = [[word.strip() for word in sentence.split(' ')] for sentence in lines if len(sentence) != 0]
    # 构建词典
    vocab = Vocabulary.build(text, min_freq=min_freq, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    print(f'vocab size:{len(vocab)}')
    # 构建语料:将单词转换为ID
    corpus = [vocab.to_ids(sentence) for sentence in text]

    return corpus, vocab


def save_pretrained(vocab, embeddings, save_path):
    '''
    保存预训练的模型(Word2vec或GloVe)的权重
    第一行指定了标记数和嵌入维度
    然后每一行代表一个标记的向量
    '''
    embeddings = embeddings.to_cpu().data
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f'{embeddings.shape[0]} {embeddings.shape[1]}\n')
        for idx, token in enumerate(vocab.id2token):
            vec = " ".join([f"{x:.4f}" for x in embeddings[idx]])
            f.write(f'{token} {vec}\n')
    print(f'Pretrained embeddings saved to: {save_path}')


def load_pretrained(load_path):
    '''
    加载保存的权重
    '''
    with open(load_path, 'r', encoding='utf-8') as f:
        # 读取首行->拆分->转换为int
        n, d = [int(x) for x in f.readline().split()]
        print(f'tokens number:{n}, embedding_dim:{d}')
        tokens = []
        embeds = []
        for line in f:
            line = line.rstrip().split(" ")
            # 单词 嵌入向量
            token, embed = line[0], [float(x) for x in line[1:]]
            tokens.append(token)
            embeds.append(embed)

        # 构建词表
        vocab = Vocabulary(tokens)
        embeds = Tensor(embeds)

    return vocab, embeds


def find_nearest(key, vocab, embedding_weights, top_k=3):
    idx = vocab[key]

    embed = embedding_weights[idx]
    embed = embed.reshape((-1, 1))

    score = embedding_weights @ embed
    score = score.squeeze()

    count = 0
    for i in (-score).argsort():
        i = i.item()
        if vocab.token(i) == key:
            continue
        print('{0}: {1}'.format(vocab.token(i), score[i]))
        count += 1
        if count == top_k:
            break


def search(search_key, embeddings, vocab):
    embeddings = embeddings.data
    s = np.sqrt((embeddings * embeddings).sum(1))
    embeddings /= s.reshape((s.shape[0], 1))  # normalize
    find_nearest(search_key, vocab, embeddings, top_k=3)
