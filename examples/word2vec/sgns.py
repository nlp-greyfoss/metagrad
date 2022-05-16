import numpy as np
from tqdm import tqdm

from metagrad import Tensor
from utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from metagrad.dataset import Dataset
import metagrad.functions as F


class SGNSDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2, n_negatives=5, ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 模型输入：(w, context)
                # 输出：0/1，表示context是否为负样本
                w = sentence[i]
                left_context_index = max(0, i - window_size)
                right_context_index = min(len(sentence), i + window_size)
                context = sentence[left_context_index:i] + sentence[i + 1:right_context_index + 1]
                context += [self.pad] * (2 * window_size - len(context))
                self.data.append((w, context))

        # 负样本数量
        self.n_negatives = n_negatives
        # 负采样分布：若参数ns_dist为None，则使用uniform分布
        self.ns_dist = ns_dist if ns_dist is not None else Tensor.ones(len(vocab))

        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = Tensor([ex[0] for ex in examples])
        contexts = Tensor([ex[1] for ex in examples])

        batch_size, window_size = contexts.shape
        neg_contexts = []
        # 对batch内的样本分别进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前样本中的context
            ns_dist = self.ns_dist.index_fill_(0, contexts[i], .0)
            neg_contexts.append(multinomial(ns_dist, self.n_negatives * window_size, replacement=True))
        neg_contexts = F.stack(neg_contexts, axis=0)
        return words, contexts, neg_contexts
