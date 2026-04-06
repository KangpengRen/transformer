from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple

import torch
from torch.utils.data import Dataset

# 特殊符号
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def normalize_text(s: str) -> str:
    """
    对数据进行简单清洗

    Args:
        s (str): 原始输入文本

    Returns:
        str: 清洗后文本
    """

    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = re.sub(r"([.!?,:\'()\-;])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def tokenize(s: str) -> List[str]:
    """
    tokenize编码

    Args:
        s (str): 原始输入文本

    Returns:
        List[str]: tokenize后的结果
    """
    return normalize_text(s).split(" ")


@dataclass
class Vocab:
    """
    词表类，包含了stoi（字符到id的映射）和itos（id到字符的映射），以及一些特殊符号的id
    """

    stoi: Dict[str, int]  # 字符到id的映射
    itos: List[str]  # id到字符的映射
    pad_id: int  # 填充符号的id
    bos_id: int  # 句子开始符号的id
    eos_id: int  # 句子结束符号的id
    unk_id: int  # 未知符号的id

    def encode(self, tokens: List[str]) -> List[int]:
        """
        编码，将字符映射为id列表

        Args:
            tokens (List[str]): 原始句子字符列表

        Returns:
            List[int]: 映射后句子的id列表
        """
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> List[str]:
        """
        解码，将id列表映射为字符列表

        Args:
            ids (List[int]): 句子的id列表
            stop_at_eos (bool, optional): 是否在遇到eos符号时停止. Defaults to True.

        Returns:
            List[str]: 映射后句子的字符列表
        """
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            out.append(self.itos[i] if 0 <= i <= len(self.itos) else UNK)
        return out


def build_vocab(
    tokenized_sentences: Iterable[List[str]], min_freq: int = 2, max_size: int = 20000
) -> Vocab:
    """
    创建词表

    Args:
        tokenized_sentences (Iterable[List[str]]): 学习词表的语料库
        min_freq (int, optional): 最小词频，小于最小的词不纳入统计. Defaults to 2.
        max_size (int, optional): 词表最大大小. Defaults to 20000.

    Returns:
        Vocab: 构建出的词表
    """
    from collections import Counter

    counter = Counter()
    for toks in tokenized_sentences:
        counter.update(toks)
    itos = [PAD, BOS, EOS, UNK]
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in (PAD, BOS, EOS, UNK):
            continue
        itos.append(tok)
        if len(itos) >= max_size:
            break

    stoi = {tok: i for i, tok in enumerate(itos)}

    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi[PAD],
        bos_id=stoi[BOS],
        eos_id=stoi[EOS],
        unk_id=stoi[UNK],
    )


def read_parallel_tsv(path: str, max_pairs: int = None) -> List[Tuple[str, str]]:
    """
    读取文件

    Args:
        path (str): 文件路径
        max_pairs (int, optional): 最大对数. Defaults to None.

    Returns:
        List[Tuple[str, str]]: 读取的句子对列表
    """
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue

            en, fr = cols[0], cols[1]
            pairs.append((en, fr))
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


def filter_by_len(pairs: List[Tuple[str, str]], max_len: int) -> List[Tuple[str, str]]:
    """
    过滤掉过长的句子对

    Args:
        pairs (List[Tuple[str, str]]): 原始句子对列表
        max_len (int): 最大长度，超过这个长度的句子对会被过滤掉

    Returns:
        List[Tuple[str, str]]: 过滤后的句子对列表
    """
    kept = []
    for s, t in pairs:
        s_tok = tokenize(s)
        t_tok = tokenize(t)
        if len(s_tok) <= max_len and len(t_tok) <= max_len:
            kept.append((s_tok, t_tok))
    return kept


class TranslationDataset(Dataset):
    """
    翻译数据集
    """

    def __init__(
        self,
        data: List[Tuple[List[str], List[str]]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
    ):
        """
        翻译数据集初始化

        Args:
            data (List[Tuple[List[str], List[str]]]): 数据列表，元素为(src_tokens, tgt_tokens)的二元组
            src_vocab (Vocab): 源语言词表
            tgt_vocab (Vocab): 目标语言词表
        """
        super().__init__()
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        """
        获取数据集长度

        Returns:
            int: 数据集长度
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集元素

        Args:
            index (int): 元素索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 源语言id序列和目标语言id序列的二元组
        """
        src_tokens, tgt_tokens = self.data[index]
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = (
            [self.tgt_vocab.bos_id]
            + self.tgt_vocab.encode(tgt_tokens)
            + [self.tgt_vocab.eos_id]
        )
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], src_pad_id: int, tgt_pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将句子对批次化，并进行填充

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): 批次数据
        src_pad_id (int): 源语言填充id
        tgt_pad_id (int): 目标语言填充id

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 填充后的批次数据
    """
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(x) for x in src_seqs]
    tgt_lens = [len(x) for x in tgt_seqs]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    batch_size = len(batch)
    src = torch.full((batch_size, max_src), src_pad_id, dtype=torch.long)
    tgt = torch.full((batch_size, max_tgt), tgt_pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src[i, : len(s)] = s
        tgt[i, : len(t)] = t
    return src, tgt
