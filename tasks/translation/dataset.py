from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple

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
    s = re.sub(r"([.!?,:\'()\-\:;])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def tokenize(s: str):
    return normalize_text(s).split(" ")


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    def encode(self, tokens):
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids, stop_at_eos: bool = True):
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


def read_parallel_tsv(path, max_pairs=None):
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


def flitter_by_len(pairs: List[Tuple[str, str]], max_len: int) -> List[Tuple[str, str]]:
    kept = []
    for s, t in pairs:
        s_tok = tokenize(s)
        t_tok = tokenize(t)
        if len(s_tok) <= max_len and len(t_tok) <= max_len:
            kept.append((s_tok, t_tok))
    return kept


class TranslationDataset(Dataset):

    def __init__(self, data, src_vocab: Vocab, tgt_vocab: Vocab):
        super().__init__()
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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


def collate_fn(batch, src_pad_id, tgt_pad_id):
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
