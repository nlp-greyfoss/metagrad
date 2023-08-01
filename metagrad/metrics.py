import collections
import math
import numpy as np

from metagrad import Tensor
from metagrad.utils import ngrams_iterator
from typing import List


def _compute_ngram_counter(tokens, max_n):
    ngrams_counter = collections.Counter(tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter


def bleu_score2(pred_seq, label_seq, k=4):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def bleu_score(candidate_corpus, references_corpus, max_n: int = 4,
               weights=[0.25] * 4) -> float:
    clipped_counts = np.zeros(max_n)
    total_counts = np.zeros(max_n)
    weights = np.array(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        refs_len_list = [float(len(ref)) for ref in refs]
        # 得到与candidate在长度上最近的ref长度
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        references_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            references_counters = references_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & references_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * np.log(pn)
        score = np.exp(log_pn.sum())

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()


if __name__ == '__main__':
    # tokens = ['me', 'me', 'you']
    # print(_compute_ngram_counter(tokens, 2))
    candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
    references_corpus = [[['My', 'full', 'pytorch', 'test']], [['No', 'Match']]]
    print(bleu_score(candidate_corpus, references_corpus))
    # print(bleu_score("你 敢 ！", "你 敢 ！", k=4))
