import numpy as np
from hanziconv import HanziConv

from examples.embeddings.utils import Vocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from metagrad import Tensor
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.utils import pad_sequence


def cht_to_chs(sent):
    # pip install hanziconv
    # 繁体转换为简体
    sent = HanziConv.toSimplified(sent)
    sent.encode("utf-8")
    return sent


def read_nmt(file_path='../data/en-cn/train_mini.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def process_nmt(text):
    """预处理“英文－中文”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格，并全部转换为小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英文－中文”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))  # 英文按空格切分
            target.append([char for char in parts[1]])  # 中文按字切分
    return source, target


# raw_text = cht_to_chs(read_nmt())
# print(raw_text[:74])

# text = process_nmt(raw_text)
# print(text[:76])

from pprint import pprint


# source, target = tokenize_nmt(text, 6)
# pprint(source)
# pprint(target)

# min_freq = 1
#
# src_vocab = Vocabulary.build(source, min_freq=min_freq, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])


# print(len(src_vocab))
# print(src_vocab.token(0))
# print(src_vocab.token(1))


def truncate_pad(line, max_len, padding_token):
    """截断或填充文本序列"""
    if len(line) > max_len:
        return line[:max_len]  # 截断
    return line + [padding_token] * (max_len - len(line))  # 填充


def build_array_nmt(lines, vocab, max_len=None):
    if not max_len:
        max_len = max(len(x) for x in lines)

    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab[EOS_TOKEN]] for l in lines]
    return np.array([truncate_pad(l, max_len, vocab[PAD_TOKEN]) for l in lines])


# print(truncate_pad(src_vocab[source[0]], 10, src_vocab[PAD_TOKEN]))

#
# src_array = build_array_nmt(source, src_vocab, 10)
# print(src_array)


class NMTDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(examples):
        src = [Tensor(ex[0]) for ex in examples]
        tgt = [Tensor(ex[1]) for ex in examples]
        # 这里只是转换为Tensor向量
        src = pad_sequence(src)
        tgt = pad_sequence(tgt)
        return src, tgt


def load_dataset_nmt(data_path, batch_size=32, min_freq=1, max_len=20):
    # 读取原始文本
    raw_text = read_nmt(data_path)
    # 处理英文符号
    text = process_nmt(raw_text)
    # 中英文分词
    source, target = tokenize_nmt(text)
    # 构建源和目标词表
    src_vocab = Vocabulary.build(source, min_freq=min_freq, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    tgt_vocab = Vocabulary.build(target, min_freq=min_freq, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    print(f'Source vocabulary size: {len(src_vocab)}, Target vocabulary size: {len(tgt_vocab)}')
    # 转换成批数据
    src_array = build_array_nmt(source, src_vocab, max_len)
    tgt_array = build_array_nmt(target, tgt_vocab, max_len)
    # 构建数据集
    dataset = NMTDataset([(src, tgt) for src, tgt in zip(src_array, tgt_array)])
    # 数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=True)
    # 返回加载器和两个词表
    return data_loader, src_vocab, tgt_vocab


#
# train_dataset, src_vocab, tgt_vocab = load_dataset_nmt('../data/en-cn/train_mini.txt')
# for X, Y in train_dataset:
#     print('X:', X.shape)
#     print('Y:', Y.shape)
#     break
