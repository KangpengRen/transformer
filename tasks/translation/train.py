from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.append(ROOT)

from transformer_core import OriginalTransformer, TransformerConfig
from data_unzip import prepare_dataset
from dataset import (
    read_parallel_tsv,
    filter_by_len,
    build_vocab,
    TranslationDataset,
    collate_fn,
    tokenize,
)


def set_seed(seed: int):
    """
    设置随机种子

    Args:
        seed (int): 种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_config(args) -> TransformerConfig:
    """
    生成配置信息

    Args:
        args (_type_): 配置参数

    Returns:
        TransformerConfig: Transformer配置文件
    """
    return TransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm_first=args.norm_first,
        pos_type=args.pos_type,
        seq_max_len=args.max_len + 50,
    )


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src_ids: torch.Tensor,
    src_pad_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """

    Args:
        model (nn.Module): Transformer模型
        src_ids (torch.Tensor): 训练输入的源语言id序列，形状为(batch_size, src_seq_len)
        src_pad_mask (torch.Tensor): 源语言id序列的padding掩码，形状为(batch_size, src_seq_len)，其中True表示对应位置是padding
        bos_id (int): 目标语言序列开始标记的id
        eos_id (int): 目标语言序列结束标记的id
        pad_id (int): 目标语言序列填充标记的id
        max_new_tokens (int):  生成的最大新token数量
        device (torch.device): 设备

    Returns:
        torch.Tensor: 生成的目标语言id序列，形状为(batch_size, gen_seq_len)，其中gen_seq_len <= max_new_tokens + 1
    """
    model.eval()  # 设置模型为评估模式

    batch_size = src_ids.size(0)
    # 初始化生成的目标语言序列，初始时只有开始标记
    tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        tgt_pad = tgt.eq(pad_id)
        # 调用模型进行前向传播，获取下一个token的预测结果
        logits = model(
            src_token_ids=src_ids,
            tgt_token_ids=tgt,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad,
        )
        # 从预测结果中选择概率最高的token作为下一个token
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # 将下一个token添加到生成的目标语言序列中
        tgt = torch.cat([tgt, next_id], dim=1)
        # 如果生成的下一个token都是结束标记，则停止生成
        if torch.all(next_id.squeeze(1).eq(eos_id)):
            break
    return tgt


def train():
    """
    训练函数，包含数据准备、模型构建、训练循环、验证和模型保存等步骤
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pairs", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=20, help="句子最大token数量")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--vocab_size", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epoches", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--pos_type",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "abs", "none"],
    )

    ap.add_argument("--norm_first", action="store_true")

    ap.add_argument("--max_len_pos", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_name", type=str, default="transformer_translation.py")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # 1. 准备数据
    data_txt = prepare_dataset(os.path.join(HERE, "data"))
    pairs_raw = read_parallel_tsv(data_txt, max_pairs=args.max_pairs)
    pairs_tok = filter_by_len(pairs_raw, max_len=args.max_len)
    random.shuffle(pairs_tok)
    n = len(pairs_tok)
    n_train = int(n * 0.9)
    train_data = pairs_tok[:n_train]
    val_data = pairs_tok[n_train:]

    # 2. 构建词表
    src_vocab = build_vocab(
        (s for s, _ in train_data), min_freq=args.min_freq, max_size=args.vocab_size
    )
    tgt_vocab = build_vocab(
        (t for _, t in train_data), min_freq=args.min_freq, max_size=args.vocab_size
    )
    train_ds = TranslationDataset(train_data, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_ds = TranslationDataset(val_data, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(
            batch=b, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(
            batch=b, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id
        ),
    )

    # 3. 构建模型
    config = make_config(args=args)
    model = OriginalTransformer(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        config=config,
        tie_embeddings=False,
    ).to(device)

    # 4. 构建优化器与损失函数
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    # 5. 训练
    best_val = float("inf")  # 验证集上最好的损失，用于保存最佳模型
    # 模型检查点路径，保存最佳模型参数和相关信息
    ckpt_path = os.path.join(HERE, "checkpoints", args.save_name)

    for epoch in range(1, args.epoches + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for step, (src_ids, tgt_ids) in enumerate(train_loader, start=1):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_in = tgt_ids[:, :-1]  # 输入的目标语言id序列，去掉最后一个token
            tgt_out = tgt_ids[:, 1:]  # 输出的目标语言id序列，去掉第一个token

            src_pad = src_ids.eq(src_vocab.pad_id)
            tgt_pad = tgt_in.eq(tgt_vocab.pad_id)

            logits = model(
                src_token_ids=src_ids,
                tgt_token_ids=tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
                tgt_causal_mask=None,
            )

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            with torch.no_grad():
                non_pad = tgt_out.ne(tgt_vocab.pad_id).sum().item()

            total_loss += loss.item() * max(non_pad, 1)
            total_tokens += max(non_pad, 1)

            if step % 100 == 0:
                print(
                    f"[train] epoch={epoch}, step={step}, loss={total_loss/total_tokens:.4f}"
                )

        train_loss = total_loss / total_tokens

        # 6. 验证
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0

        with torch.no_grad():
            for src_ids, tgt_ids in val_loader:
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)

                tgt_in = tgt_ids[:, :-1]
                tgt_out = tgt_ids[:, 1:]

                src_pad = src_ids.eq(src_vocab.pad_id)
                tgt_pad = tgt_in.eq(tgt_vocab.pad_id)

                logits = model(
                    src_token_ids=src_ids,
                    tgt_token_ids=tgt_in,
                    src_key_padding_mask=src_pad,
                    tgt_key_padding_mask=tgt_pad,
                    tgt_causal_mask=None,
                )

                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                non_pad = tgt_out.ne(tgt_vocab.pad_id).sum().item()
                val_loss_sum += loss.item() * max(non_pad, 1)
                val_tokens += max(non_pad, 1)

        val_loss = val_loss_sum / val_tokens
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # 7. 保存最佳模型
        if val_loss < best_val:
            best_val = val_loss
            payload = {
                "model_state": model.state_dict(),
                "config": asdict(config) if hasattr(config, "__dict__") else None,
                "src_vocab": {
                    "itos": src_vocab.itos,
                    "pad_id": src_vocab.pad_id,
                    "bos_id": src_vocab.bos_id,
                    "eos_id": src_vocab.eos_id,
                    "unk_id": src_vocab.unk_id,
                },
                "tgt_vocab": {
                    "itos": tgt_vocab.itos,
                    "pad_id": tgt_vocab.pad_id,
                    "bos_id": tgt_vocab.bos_id,
                    "eos_id": tgt_vocab.eos_id,
                    "unk_id": tgt_vocab.unk_id,
                },
                "args": vars(args),
            }
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(payload, ckpt_path)
            print(f"[save] {ckpt_path}")

        # 8. 可视化
        demo_src = ["i am cold.", "he is my friend.", "where is the classroom?"]
        for s in demo_src:
            toks = tokenize(s)
            ids = [src_vocab.stoi.get(t, src_vocab.unk_id) for t in toks]
            src_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            src_pad = src_tensor.eq(src_vocab.pad_id)
            gen = greedy_decode(
                model=model,
                src_ids=src_tensor,
                src_pad_mask=src_pad,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                pad_id=tgt_vocab.pad_id,
                max_new_tokens=args.max_len + 5,
                device=device,
            )
            out_ids = gen[0].tolist()[1:]
            out_toks = [
                tgt_vocab.itos[i] if i < len(tgt_vocab.itos) else "<unk>"
                for i in out_ids
            ]
            if tgt_vocab.eos_id in out_ids:
                out_toks = out_toks[: out_ids.index(tgt_vocab.eos_id)]
            print(f"[demo] {s} -> {' '.join(out_toks)}")

    print("success")


if __name__ == "__main__":
    train()
