"""
分词器实现：BPE 和 WordPiece
"""

import re
import regex  # pip install regex（支持 \p{L} \p{N} 等 Unicode 类别）
from collections import defaultdict

_PRETOK_RE = re.compile(r"\w+|[^\w\s]+")  # WordPiece 用（Whitespace 风格）

# GPT-2 原版预分词正则（需要 regex 模块）
# - ' ?\p{L}+'  : 可选空格 + 连续字母（空格会被编码进 token，成为 Ġ 前缀）
# - ' ?\p{N}+'  : 可选空格 + 连续数字
# - ' ?[^\s\p{L}\p{N}]+' : 可选空格 + 其他非空白字符（标点、_、= 等）
# - '\s+(?!\S)' / '\s+' : 独立空白（行尾空格等）
_GPT2_PAT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 的字节到可打印 Unicode 字符的映射。

    ASCII 可打印字符（33-126）以及 Latin-1 可打印字符（161-172, 174-255）
    直接映射到自身；其余字节（含空格=32、控制字符等）映射到从 U+0100
    开始的 Unicode 区域，以保证词表中不出现不可打印字符。

    最重要的映射：空格（byte 32）→ 'Ġ'（U+0120）。
    """
    # 已可打印的字节直接映射到自身
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# 全局字节↔字符映射表
_BYTE2CHAR: dict[int, str] = _bytes_to_unicode()
_CHAR2BYTE: dict[str, int] = {v: k for k, v in _BYTE2CHAR.items()}
_SPACE_CHAR = _BYTE2CHAR[ord(" ")]  # 'Ġ'（U+0120）


# ─────────────────────────────────────────────
#  BPE Tokenizer
# ─────────────────────────────────────────────


class BPETokenizer:
    """
    字节级别 BPE（Byte-level BPE）分词器。

    与原始 Sennrich 2016 BPE 的区别：
    - 初始词表：256 字节对应的可打印 Unicode 字符（GPT-2 映射），不再按语料扫描；
    - 词边界：用 Ġ（空格的字节映射，U+0120）作为词头**前缀**，而非 </w> 后缀；
    - 无 <unk>：任意输入均可拆为字节，不会产生未登录词。

    算法思路：
    1. 预分词：用正则 \\w+|[^\\w\\s]+ 切分，若某 token 前有空格则加 Ġ 前缀；
    2. 统计相邻子词对的出现频次；
    3. 合并频次最高的子词对，重复直到达到目标词表大小。
    """

    def __init__(self, vocab_size: int, lower_case: bool = False):
        self.vocab_size = vocab_size
        self.lower_case = lower_case
        self.merges: list[tuple] = []  # 按顺序记录的合并规则
        self.vocab: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    # ── 预分词 ────────────────────────────────

    def _pretokenize(self, text: str) -> list[str]:
        """
        GPT-2 式预分词：用 GPT-2 正则切分后，把每个 pre-token（含可选前驱空格）
        编码为 UTF-8 字节序列，再通过 bytes_to_unicode 映射为可打印字符串。

        例："def forward" → ['def', 'Ġforward']
             "你好"       → ['ä½ ', 'å¥½']   （UTF-8 字节的可打印表示）
        """
        if self.lower_case:
            text = text.lower()
        return [
            "".join(_BYTE2CHAR[b] for b in token.encode("utf-8"))
            for token in _GPT2_PAT.findall(text)
        ]

    # ── 训练 ──────────────────────────────────

    def _get_word_freqs(self, corpus: list[str]) -> dict:
        """将语料库中的 pre-token 拆成字符元组，统计频次。"""
        word_freqs: dict[tuple, int] = defaultdict(int)
        for text in corpus:
            for word in self._pretokenize(text):
                word_freqs[tuple(word)] += 1
        return dict(word_freqs)

    def _get_pair_freqs(self, word_freqs: dict) -> dict:
        """统计所有单词中相邻子词对的频次。"""
        pair_freqs: dict[tuple, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        return dict(pair_freqs)

    def _merge_pair(self, word_freqs: dict, pair: tuple) -> dict:
        """将 word_freqs 中每个词里出现的 pair 合并为一个子词。"""
        merged = "".join(pair)
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus: list[str]):
        """从语料库中学习 BPE 合并规则，构建词表。"""
        word_freqs = self._get_word_freqs(corpus)

        # 初始词表：特殊 token + 256 字节对应的 GPT-2 可打印字符
        # 无需 </w>，词边界由 Ġ 前缀承载
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for i in range(256):
            ch = _BYTE2CHAR[i]
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # 迭代合并，直到达到目标词表大小
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(word_freqs, best_pair)
            self.merges.append(best_pair)
            merged_token = "".join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

        self.id2token = {i: tok for tok, i in self.vocab.items()}

    # ── 推理 ──────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """对单个 pre-token 按顺序应用所有 BPE 合并规则。"""
        symbols = list(word)  # 不加 </w>，Ġ 已在 word 头部
        for pair in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        """将文本转换为 token id 列表。"""
        unk_id = self.vocab.get("<unk>", 1)
        ids = []
        for word in self._pretokenize(text):
            for tok in self._tokenize_word(word):
                ids.append(self.vocab.get(tok, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        将 token id 列表还原为文本。

        先把 token 字符串拼接，再通过 _CHAR2BYTE 逐字符映射回字节，
        最后按 UTF-8 解码——与 GPT-2 Encoder.decode() 完全一致。
        """
        text = "".join(self.id2token.get(i, "") for i in ids)
        return bytearray(_CHAR2BYTE[c] for c in text).decode("utf-8", errors="replace")


# ─────────────────────────────────────────────
#  WordPiece Tokenizer
# ─────────────────────────────────────────────


class WordPieceTokenizer:
    """
    WordPiece 分词器（BERT 采用的分词算法）。

    与 BPE 的核心区别：
    - BPE 合并频次最高的子词对；
    - WordPiece 合并使语言模型似然最大的子词对，等价于：
        score(A, B) = freq(AB) / (freq(A) * freq(B))

    子词前缀约定：
    - 单词的首子词不加前缀；
    - 非首子词加 '##' 前缀（表示是前一个子词的延续）。

    参考：Google's Neural Machine Translation System (Wu et al., 2016)
    """

    def __init__(self, vocab_size: int, lower_case: bool = False):
        self.vocab_size = vocab_size
        self.lower_case = lower_case
        self.vocab: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # ── 训练 ──────────────────────────────────

    def _get_word_freqs(self, corpus: list[str]) -> dict:
        """将单词拆为带 ## 前缀的字符序列，统计频次。"""
        word_freqs: dict[tuple, int] = defaultdict(int)
        for text in corpus:
            for word in _PRETOK_RE.findall(text.lower() if self.lower_case else text):
                # 首字符不加前缀，后续字符加 ## 前缀
                chars = [word[0]] + ["##" + ch for ch in word[1:]]
                word_freqs[tuple(chars)] += 1
        return dict(word_freqs)

    def _get_pair_scores(self, word_freqs: dict) -> dict:
        """计算每对相邻子词的 WordPiece 得分。"""
        token_freqs: dict[str, int] = defaultdict(int)
        pair_freqs: dict[tuple, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for tok in word:
                token_freqs[tok] += freq
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        # score = freq(AB) / (freq(A) * freq(B))
        scores = {
            pair: freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def _merge_pair(self, word_freqs: dict, pair: tuple) -> dict:
        """合并 word_freqs 中所有词里出现的子词对。"""
        a, b = pair
        # b 带 ## 前缀，合并时去掉 ## 直接拼接
        merged = a + b.lstrip("#")
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus: list[str]):
        """从语料库中学习 WordPiece 词表。"""
        word_freqs = self._get_word_freqs(corpus)

        # 初始词表：特殊 token + 所有字符（含 ## 前缀的非首字符）
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for word in word_freqs:
            for ch in word:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        # 迭代合并，直到达到目标词表大小
        while len(self.vocab) < self.vocab_size:
            scores = self._get_pair_scores(word_freqs)
            if not scores:
                break
            best_pair = max(scores, key=scores.get)
            word_freqs = self._merge_pair(word_freqs, best_pair)
            a, b = best_pair
            merged = a + b.lstrip("#")
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

        self.id2token = {i: tok for tok, i in self.vocab.items()}

    # ── 推理 ──────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """用最长匹配（贪心）对单词进行子词切分。"""
        if not word:
            return []
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    found = substr
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            tokens.append(found)
            start = end
        return tokens

    def encode(self, text: str) -> list[int]:
        """将文本转换为 token id 列表。"""
        unk_id = self.vocab.get("[UNK]", 1)
        ids = []
        for word in _PRETOK_RE.findall(text.lower() if self.lower_case else text):
            for tok in self._tokenize_word(word):
                ids.append(self.vocab.get(tok, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        """将 token id 列表转换回文本（合并 ## 前缀的子词）。"""
        tokens = [self.id2token.get(i, "[UNK]") for i in ids]
        text = ""
        for tok in tokens:
            if tok.startswith("[") or tok not in self.vocab:
                text += " " + tok
            elif tok.startswith("##"):
                text += tok[2:]
            else:
                text += " " + tok
        return text.strip()


# ─────────────────────────────────────────────
#  测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    corpus = [
        "low lower newest wildest",
        "low lower lower",
        "new new lower lowest",
        "wildest wild wild",
    ]

    print("=" * 40)
    print("BPE Tokenizer")
    print("=" * 40)
    # 初始词表已有 256 字节字符 + 4 个特殊 token = 260 个，vocab_size 必须 > 260 才会有合并
    bpe = BPETokenizer(vocab_size=300)
    bpe.train(corpus)
    print(f"词表大小:   {len(bpe.vocab)}")
    print(f"合并规则数: {len(bpe.merges)}")
    print(f"前10条合并: {bpe.merges[:10]}")

    text = "lower new wild"
    ids = bpe.encode(text)
    print(f"\n输入: {text!r}")
    print(f"编码: {ids}")
    print(f"解码: {bpe.decode(ids)!r}")

    print()
    print("=" * 40)
    print("WordPiece Tokenizer")
    print("=" * 40)
    wp = WordPieceTokenizer(vocab_size=50)
    wp.train(corpus)
    print(f"词表大小: {len(wp.vocab)}")

    ids = wp.encode(text)
    print(f"\n输入: {text!r}")
    print(f"编码: {ids}")
    print(f"解码: {wp.decode(ids)!r}")
