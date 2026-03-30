from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass


class TokenEmbedding(nn.Module):
    """
    词嵌入层，将词表维度的token id转化为模型维度的词嵌入向量
    """

    def __init__(self, vocab_size: int, d_model):
        """
        词嵌入层初始化
        Args:
            vocab_size (int): 词表大小
            d_model (_type_): 模型维度
        """
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        测嵌入层前向传播
        Args:
            token_ids (torch.Tensor): token id，维度为vocab_size

        Returns:
            torch.Tensor: 词嵌入向量，维度为d_model
        """
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    正余弦位置编码，将位置信息编码到词嵌入向量中
    """

    def __init__(self, d_model: int, seq_max_len: int):
        """
        正余弦位置编码初始化
        Args:
            d_model (int): 模型维度
            seq_max_len (int): 最大句子长度
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.seq_max_len = seq_max_len

        # 初始化位置矩阵，维度为 (seq_max_len, d_model)
        pe = torch.zeros(seq_max_len, d_model)
        # 设置每个token的索引位置
        position = torch.arange(start=0, end=seq_max_len).unsqueeze(1)
        div_item = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_item)
        pe[:, 1::2] = torch.cos(position * div_item)

        # 将位置编码矩阵注册到缓存中，不参与后续训练
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, token_embed: torch.Tensor) -> torch.Tensor:
        """
        正余弦位置编码前向传播
        Args:
            token_embed (torch.Tensor): 词嵌入向量

        Returns:
            torch.Tensor: 附带位置编码的词嵌入向量
        """
        _, seq_len, _ = token_embed.shape
        if seq_len > self.seq_max_len:
            raise ValueError(
                f"The input value exceeds({seq_len}) the maximum limit of [{self.seq_max_len}] characters."
            )
        return token_embed + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """
    多头注意力
    """

    def __init__(self, d_model: int, num_heads: int, drop_out: float = 0):
        """
        初始化多头注意力
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            drop_out(float): 神经元随机失活概率
        """
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model({d_model} must be devisable by num_heads({num_heads})!)"
            )
        self.d_model = d_model  # 记录模型维度
        self.num_heads = num_heads  # 记录注意力头数
        self.d_head = d_model // num_heads  # 记录注意力头维度

        self.proj_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_out = nn.Linear(in_features=d_model, out_features=d_model)

        self.proj_drop = nn.Dropout(drop_out)
        self.attn_drop = nn.Dropout(drop_out)

    def forward(
        self,
        query: Optional[torch.Tensor],
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        多头注意力前向传播
        Args:
            query (Optional[torch.Tensor]): query向量
            key (Optional[torch.Tensor]): key向量
            value (Optional[torch.Tensor]): value向量
            attn_mask (Optional[torch.Tensor]): 注意力掩码矩阵
            key_padding_mask (Optional[torch.Tensor]): 词嵌入掩码矩阵
            need_weights (bool, optional): 是否需要返回权重 Defaults to False.

        Returns:
            _type_: 前向传播结果
        """

        if key is None:
            key = query
        if value is None:
            value = key

        # 将query、key、value经过线性投影
        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(value)

        batch_size, len_q, _ = q.shape
        _, len_k, _ = k.shape

        # 拆分为多头
        # (batch_size, num_heads, len_q, d_head)
        q = q.view(batch_size, len_q, self.num_heads, self.d_head).transpose(1, 2)
        # (batch_size, num_heads, len_k, d_head)
        k = k.view(batch_size, len_k, self.num_heads, self.d_head).transpose(1, 2)
        # (batch_size, num_heads, len_k, d_head)
        v = v.view(batch_size, len_k, self.num_heads, self.d_head).transpose(1, 2)

        # 计算注意力得分 (batch_size, num_head, len_q, len_k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        # 构建联合掩码
        combined_mask = build_combined_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch_size=batch_size,
            num_heads=self.num_heads,
            len_q=len_q,
            len_k=len_k,
            device=scores.device,
        )
        if combined_mask is not None:
            scores = scores.masked_fill(combined_mask, float("-inf"))

        attn = F.softmax(scores, -1)  # (batch_size, num_head, len_q, len_k)
        attn = self.attn_drop(attn)

        # 计算对token的注意力
        attn = attn @ v  # (batch_size, num_head, len_q, d_head)
        # 将注意力矩阵恢复成原本形状
        attn = attn.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model)

        # 经过线性层融合
        out = self.proj_out(attn)
        out = self.proj_drop(out)

        if need_weights:
            return out, attn
        return out


def build_combined_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    batch_size: int,
    num_heads: int,
    len_q: int,
    len_k: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    构建联合掩码
    Args:
        attn_mask (Optional[torch.Tensor]): 注意力掩码矩阵，期望维度为(*, *, len_q, len_k)
        key_padding_mask (Optional[torch.Tensor]): 词嵌入掩码矩阵，期望维度为(batch_size, len_q)
        batch_size (int): 批量大小
        num_heads (int): 注意力头数
        len_q (int): q向量长度
        len_k (int): k向量长度
        device (torch.device): 设备

    Raises:
        ValueError: 掩码矩阵维度无法识别

    Returns:
        Optional[torch.Tensor]: 联合掩码
    """
    combined = None

    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
        combined = _expand_attn_mask(
            attn_mask=attn_mask, batch_size=batch_size, num_heads=num_heads
        )
        if combined.size(-1) != len_k or combined.size(-2) != len_q:
            raise ValueError(
                f"The shapes of the attn_mask matrices({attn_mask.shape}) are incompatible for the operation. "
                f"Excepted: [{batch_size}, {num_heads}, {len_q}, {len_k}]"
            )

    if key_padding_mask is not None:
        if key_padding_mask.dim() != 2 or key_padding_mask.size(1) != len_k:
            raise ValueError(
                f"The shapes of the key_padding_mask matrices({key_padding_mask.shape}) are incompatible for the operation. "
                f"Excepted: [{batch_size}, {len_k}]"
            )
        # 补充维度
        key_padding_mask = key_padding_mask.to(device).unsqueeze(1).unsqueeze(1)
        key_padding_mask = key_padding_mask.expand(batch_size, num_heads, len_q, len_k)
        combined = (
            key_padding_mask if combined is None else (combined | key_padding_mask)
        )

    return combined


def _expand_attn_mask(
    attn_mask: Optional[torch.Tensor], batch_size: int, num_heads: int
) -> Optional[torch.Tensor]:
    """
    扩展注意力掩码
    Args:
        attn_mask (Optional[torch.Tensor]): 注意力掩码
        batch_size (int): 批量大小
        num_heads (int): 注意力头数

    Raises:
        ValueError: 注意力矩阵维度无法识别

    Returns:
        Optional[torch.Tensor]: 维度为(batch_size, num_heads, len_q, len_k)的注意力掩码矩阵
    """

    d = attn_mask.dim()
    if d == 2:
        # (len_q, len_k) -> (1, 1, len_q, len_k)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    elif d == 3:
        # (batch_size, len_q, len_k) -> (batch_size, 1, len_q, len_k)
        attn_mask = attn_mask.unsqueeze(1)
    elif d == 4:
        # (batch_size, num_heads, len_q, len_k) -> 无需修改
        pass
    else:
        raise ValueError(
            f"Unsupported attn_mask dimension: {d}, shape: {attn_mask.shape}"
        )

    # 扩展 batch 维度（如果掩码的第一维为1，而目标 batch_size 不为1）
    if attn_mask.size(0) == 1 and batch_size != 1:
        attn_mask = attn_mask.expand(batch_size, -1, -1, -1)
    # 扩展 heads 维度（如果掩码的第二维为1，而目标 num_heads 不为1）
    if attn_mask.size(1) == 1 and num_heads != 1:
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)

    return attn_mask


class FeedForward(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self, d_model: int, d_ff: int, drop_out: float = 0):
        """
        初始化前馈神经网络
        Args:
            d_model (int): 模型维度
            d_ff (int): 中间维度
            drop_out (float, optional): 神经元随机失活概率 Defaults to 0.
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.drop2 = nn.Dropout(drop_out)

    def forward(self, x):
        """
        前馈神经网络前向传播
        Args:
            x (_type_): 输入

        Returns:
            _type_: 前向传播结果
        """
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder块
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        drop_out: float,
        norm_first: bool = False,
    ):
        """
        初始化Transformer Encoder块
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            d_ff (int): 前馈神经网络维度
            drop_out (float): 神经元随机失活概率
            norm_first (bool, optional): 是否先归一化 Defaults to False.
        """
        super(TransformerEncoderLayer, self).__init__()

        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, drop_out=drop_out
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, drop_out=drop_out)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop_path1 = nn.Dropout(drop_out)
        self.drop_path2 = nn.Dropout(drop_out)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ):
        """
        Transformer Encoder块前向传播
        Args:
            x (torch.Tensor): 输入
            attn_mask (Optional[torch.Tensor]): 注意力掩码矩阵
            key_padding_mask (Optional[torch.Tensor]): 词嵌入掩码矩阵

        Returns:
            _type_: 前向传播结果
        """
        if self.norm_first:
            x = x + self.drop_path1(
                self.self_attn(
                    query=self.norm1(x),
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop_path2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self.drop_path1(
                    self.self_attn(
                        query=x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
                    )
                )
            )
            x = self.norm2(x + self.drop_path2(self.ffn(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder块
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        drop_out: float,
        norm_first: bool = False,
    ):
        """
        初始化Transformer Decoder块
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            d_ff (int): 前馈神经网络维度
            drop_out (float): 神经元随机失活概率
            norm_first (bool, optional): 是否先归一化 Defaults to False.
        """
        super(TransformerDecoderLayer, self).__init__()

        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, drop_out=drop_out
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, drop_out=drop_out
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, drop_out=drop_out)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop_path1 = nn.Dropout(drop_out)
        self.drop_path2 = nn.Dropout(drop_out)
        self.drop_path3 = nn.Dropout(drop_out)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ):
        """
        Transformer Encoder块前向传播
        Args:
            x (torch.Tensor): 输入
            memory (torch.Tensor): 编码器输出
            tgt_attn_mask (Optional[torch.Tensor]): 自注意力掩码矩阵
            tgt_key_padding_mask (Optional[torch.Tensor]): 自词嵌入掩码矩阵
            memory_key_padding_mask (Optional[torch.Tensor]): 交叉词嵌入掩码矩阵

        Returns:
            _type_: 前向传播结果
        """
        if self.norm_first:
            # 自注意力（使用 drop_path1）
            x = x + self.drop_path1(
                self.self_attn(
                    query=self.norm1(x),
                    attn_mask=tgt_attn_mask,
                    key_padding_mask=tgt_key_padding_mask,
                )
            )
            # 交叉注意力（使用 drop_path2）
            x = x + self.drop_path2(
                self.cross_attn(
                    query=self.norm2(x),
                    key=memory,
                    key_padding_mask=memory_key_padding_mask,
                )
            )
            # 前馈网络（使用 drop_path3）
            x = x + self.drop_path3(self.ffn(self.norm3(x)))
        else:
            # 自注意力 + 残差
            x = self.norm1(
                x
                + self.drop_path1(
                    self.self_attn(
                        query=x,
                        attn_mask=tgt_attn_mask,
                        key_padding_mask=tgt_key_padding_mask,
                    )
                )
            )
            # 交叉注意力 + 残差（使用正确的 padding mask）
            x = self.norm2(
                x
                + self.drop_path2(
                    self.cross_attn(
                        query=x,
                        key=memory,
                        key_padding_mask=memory_key_padding_mask,  # 修正
                    )
                )
            )
            # 前馈网络 + 残差（修正语法）
            x = self.norm3(x + self.drop_path3(self.ffn(x)))
        return x


@dataclass
class TransformerConfig:
    """
    Transformer配置类
    """

    d_model: int = 512  # 模型维度
    num_heads: int = 8  # 注意力头数
    d_ff: int = 2048  # 前馈神经网络维度
    num_layers: int = 6  # 编码器/解码器层数
    drop_out: float = 0.1  # 神经元随机失活概率

    norm_first: bool = False  # 是否先归一化
    pos_type: str = "sinusoidal"  # 位置编码方式
    max_seq_len: int = 512  # 最大长度


def _build_position(pos_type: str, max_seq_len: int, d_model: int):
    """
    获取指定类型的位置编码其
    Args:
        pos_type (str): 编码方式
        max_seq_len (int): 最大序列长度
        d_model (int): 模型维度

    Raises:
        ValueError: 无法识别的编码类型

    Returns:
        _type_: 编码器
    """
    if pos_type == "sinusoidal":
        return SinusoidalPositionalEncoding(max_seq_len, d_model)
    elif pos_type == "none":
        return None
    else:
        raise ValueError(f"Unknown positional encoding({pos_type}).")


class EncoderBone(nn.Module):
    """
    编码器骨架
    """

    def __init__(self, config: TransformerConfig):
        """
        初始化编码器骨架
        Args:
            config (TransformerConfig): Transformer配置类
        """

        super(EncoderBone, self).__init__()

        self.config = config
        self.pos = _build_position(
            pos_type=config.pos_type,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
        )
        self.drop = nn.Dropout(config.drop_out)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    drop_out=config.drop_out,
                    norm_first=config.norm_first,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_ln = (
            nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        编码器前向传播
        Args:
            x (_type_): 输入
            key_padding_mask (Optional[torch.Tensor]): 词嵌入编码

        Returns:
            Optional[torch.Tensor]: 前向传播结果
        """
        if self.pos is not None:
            x = self.pos(x)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x=x, attn_mask=None, key_padding_mask=key_padding_mask)

        x = self.final_ln(x)
        return x


class DecoderBone(nn.Module):
    """
    解码器骨架
    """

    def __init__(self, config: TransformerConfig):
        """
        初始化解码器骨架
        Args:
            config (TransformerConfig): Transformer配置类
        """

        super(DecoderBone, self).__init__()

        self.config = config
        self.pos = _build_position(
            pos_type=config.pos_type,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
        )
        self.drop = nn.Dropout(config.drop_out)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    drop_out=config.drop_out,
                    norm_first=config.norm_first,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_ln = (
            nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        编码器前向传播
        Args:
            x (_type_): 输入
            key_padding_mask (Optional[torch.Tensor]): 词嵌入编码

        Returns:
            Optional[torch.Tensor]: 前向传播结果
        """
        if self.pos is not None:
            x = self.pos(x)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        x = self.final_ln(x)
        return x


class Transformer(nn.Module):
    """
    原始Transformer模型
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: TransformerConfig,
        tie_embeddings: bool = False,
    ):
        super(Transformer, self).__init__()

        self.config = config

        self.src_token_embed = TokenEmbedding(src_vocab_size, config.d_model)
        self.tgt_token_embed = TokenEmbedding(tgt_vocab_size, config.d_model)

        self.encoder = EncoderBone(config=config)
        self.decoder = DecoderBone(config=config)

        self.generator = nn.Linear(
            in_features=config.d_model, out_features=tgt_vocab_size, bias=False
        )

        if tie_embeddings:  # 共享权重
            self.generator.weight = self.tgt_token_embed.embedding.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_causal_mask: Optional[torch.Tensor] = None,
    ):
        _, len_t = tgt_ids.shape

        if tgt_causal_mask is None:
            tgt_causal_mask = mask_causal_mask(len_t, device=src_ids.device)

        # 1. embedding
        src = self.src_token_embed(src_ids)
        tgt = self.tgt_token_embed(tgt_ids)

        # 2. encode
        memory = self.encoder(src, key_padding_mask=src_key_padding_mask)

        # 3. decode
        decoder_out = self.decoder(
            x=tgt,
            memory=memory,
            tgt_attn_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 4. generator
        logits = self.generator(decoder_out)

        return logits


def mask_causal_mask(len_t: int, device: torch.device) -> torch.Tensor:
    """
    创建因果掩码
    Args:
        len_t (int): 矩阵长度
        device (torch.device): 设备

    Returns:
        torch.Tensor: 因果掩码
    """
    return torch.triu(torch.ones(size=(len_t, len_t), device=device), diagonal=1).bool()


def _sanity_check():
    """
    检验函数
    """
    torch.manual_seed(0)

    config = TransformerConfig(
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        drop_out=0.1,
        norm_first=False,
        pos_type="sinusoidal",
        max_seq_len=128,
    )

    model = Transformer(
        src_vocab_size=1000, tgt_vocab_size=1200, config=config, tie_embeddings=False
    )
    src = torch.randint(0, 1000, (2, 16))
    tgt = torch.randint(0, 1200, (2, 12))
    src_pad = torch.zeros(2, 16).bool()
    tgt_pad = torch.zeros(2, 12).bool()

    logits = model(src, tgt, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
    assert logits.shape == (2, 12, 1200)
    print("sanity check OK")
    print(
        f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )


if __name__ == "__main__":
    _sanity_check()
