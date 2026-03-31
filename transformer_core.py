from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    词嵌入层
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        词嵌入层初始化

        Args:
            vocab_size (int): 词表大小
            d_model (int): 模型维度
        """
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        词嵌入层前向传播

        Args:
            token_ids (torch.Tensor): 词表中token id序列，形状为 (batch_size, seq_len)

        Returns:
            torch.Tensor: 词嵌入张量
        """
        return self.embed(token_ids) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    正余弦位置编码
    """

    def __init__(self, d_model: int, seq_max_len: int):
        """
        正余弦位置编码初始化

        Args:
            d_model (int): 模型维度
            seq_max_len (int): 最大序列长度
        """

        super(SinusoidalPositionalEncoding, self).__init__()
        self.seq_max_len = seq_max_len

        pe = torch.zeros(seq_max_len, d_model)  # 初始化位置矩阵 (seq_max_len, d_model)
        # 初始化索引位置 (seq_max_len, 1)
        position = torch.arange(seq_max_len).unsqueeze(1)
        div_item = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_item)
        pe[:, 1::2] = torch.cos(position * div_item)

        # 将pe矩阵注册到缓存中 (1, seq_max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, token_embed: torch.Tensor) -> torch.Tensor:
        """
        正余弦位置编码前向传播

        Args:
            token_embed (torch.Tensor): 词嵌入张量，形状为：(batch_size, seq_len, d_model)

        Raises:
            ValueError: 输入词嵌入张量长度超过最大限度

        Returns:
            torch.Tensor: 附带位置信息的词嵌入张量
        """
        _, seq_len, _ = token_embed.shape
        if seq_len > self.seq_max_len:
            raise ValueError(
                f"The input value exceeds({seq_len}) the maximum limit of [{self.seq_max_len}] characters."
            )
        return token_embed + self.pe[:, :seq_len, :]


class MultiheadAttention(nn.Module):
    """
    多头注意力
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0):
        """
        多头注意力初始化

        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            dropout (float, optional): 神经元随机失活概率 Defaults to 0.

        Raises:
            ValueError: 模型维度无法被注意力头数整除
        """
        super(MultiheadAttention, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model({d_model}) must be devisable by num_heads({num_heads})!"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.proj_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_out = nn.Linear(in_features=d_model, out_features=d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: Optional[torch.Tensor],
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        多头注意力前向传播

        Args:
            query (Optional[torch.Tensor]): query张量
            key (Optional[torch.Tensor]): key张量
            value (Optional[torch.Tensor]): value张量
            attn_mask (Optional[torch.Tensor], optional): 注意力掩码 Defaults to None.
            key_padding_mask (Optional[torch.Tensor], optional): 词嵌入掩码 Defaults to None.
            need_weights (bool, optional): 是否返回注意力权重 Defaults to False.

        Returns:
            _type_: 对token的注意力
        """
        if key is None:
            key = query
        if value is None:
            value = key

        # 获取维度信息
        batch_size, len_q, d_model = query.shape
        _, len_k, _ = key.shape

        # 将query、key、value线性投影
        q = self.proj_q(query)  # (batch_size, len_q, d_model)
        k = self.proj_k(key)  # (batch_size, len_k, d_model)
        v = self.proj_v(value)  # (batch_size, len_k, d_model)

        # 拆分为多头
        # (batch_size, num_heads, len_q, d_head)
        q = q.view(batch_size, len_q, self.num_heads, self.d_head).transpose(1, 2)
        # (batch_size, num_heads, len_k, d_head)
        k = k.view(batch_size, len_k, self.num_heads, self.d_head).transpose(1, 2)
        # (batch_size, num_heads, len_k, d_head)
        v = v.view(batch_size, len_k, self.num_heads, self.d_head).transpose(1, 2)

        # 计算注意力得分
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

        # 在最后一个维度进行softmax，得到注意力
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 注意力与v进行内积，计算最每个token的注意力
        attn_output = attn_weights @ v  # (batch_size, num_heads, len_q, len_k)
        # 恢复到单头形状
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, len_q, d_model)
        )

        # 将特征线性融合
        out = self.proj_out(attn_output)
        out = self.proj_drop(out)

        if need_weights:
            return out, attn_weights
        return out


def _expand_attn_mask(
    attn_mask: Optional[torch.Tensor],
    batch_size: int,
    num_heads: int,
) -> Optional[torch.Tensor]:
    """
    将注意力掩码矩阵扩展到需要维度 (batch_size, num_heads, len_q, len_k)

    Args:
        attn_mask (Optional[torch.Tensor]): 注意力掩码矩阵
        batch_size (int): 批量大小
        num_heads (int): 注意力头数

    Raises:
        ValueError: 注意力掩码矩阵维度异常

    Returns:
        Optional[torch.Tensor]: 扩展后的注意力掩码矩阵
    """
    # 获取因果掩码的维度，目标维度 (batch_size, num_heads, len_q, len_k)
    d = attn_mask.dim()
    if d == 2:
        # 仅有两维，补充batch_size和num_heads
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    elif d == 3:
        # 仅有三维，补充num_heads
        attn_mask = attn_mask.unsqueeze(1)
    elif d == 4:
        # 为目标维度，不进行更改
        pass
    else:
        raise ValueError(
            f"Unsupported attn_mask dimension: {d}, shape: {attn_mask.shape}"
        )

    if attn_mask.size(0) == 1 and batch_size != 1:
        # batch_size维度不匹配，进行扩充
        attn_mask = attn_mask.expand(batch_size, -1, -1, -1)
    if attn_mask.size(1) == 1 and num_heads != 1:
        # num_heads维度不匹配，进行扩充
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)

    return attn_mask


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
    根据注意力掩码矩阵和词嵌入掩码矩阵构建联合掩码

    Args:
        attn_mask (Optional[torch.Tensor]): 注意力掩码矩阵
        key_padding_mask (Optional[torch.Tensor]): 词嵌入掩码矩阵
        batch_size (int): 批量大小
        num_heads (int): 注意力头数
        len_q (int): q张量长度
        len_k (int): k张量长度
        device (torch.device): 设备

    Raises:
        ValueError: 注意力掩码矩阵维度信息异常
        ValueError: 词嵌入掩码矩阵维度信息异常

    Returns:
        Optional[torch.Tensor]: 联合掩码矩阵
    """
    combined = None

    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
        combined = _expand_attn_mask(attn_mask, batch_size, num_heads)
        if combined.size(-1) != len_k or combined.size(-2) != len_q:
            raise ValueError(
                f"The shapes of the attn_mask matrices({attn_mask.shape}) are incompatible for the operation. "
                f"Excepted: [{batch_size}, {num_heads}, {len_q}, {len_k}]"
            )

    if key_padding_mask is not None:
        # key_padding_mask维度应该为：(batch_size, len_k)
        if key_padding_mask.dim() != 2 or key_padding_mask.size(1) != len_k:
            raise ValueError(
                f"The shapes of the key_padding_mask matrices({key_padding_mask.shape}) are incompatible for the operation. "
                f"Excepted: [{batch_size}, {len_k}]"
            )
        key_padding_mask = key_padding_mask.to(device).unsqueeze(1).unsqueeze(1)
        key_padding_mask = key_padding_mask.expand(batch_size, num_heads, len_q, len_k)
        combined = key_padding_mask if combined is None else combined | key_padding_mask

    return combined


class FeedForward(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0):
        """
        前馈神经网络初始化

        Args:
            d_model (int): 模型维度
            d_ff (int): 前馈神经网络维度
            dropout (float, optional): 神经元随机失活概率 Defaults to 0.
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        前馈神经网络前向传播

        Args:
            x (Optional[torch.Tensor]): 输入张量

        Returns:
            Optional[torch.Tensor]: 输出张量
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
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        """
        Transformer Encoder初始化

        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            d_ff (int): 前馈神经网络维度
            dropout (float, optional): 神经元随机失活概率 Defaults to 0.1.
            norm_first (bool, optional): 是否先归一化 Defaults to False.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Transformer Encoder前向传播

        Args:
            x (Optional[torch.Tensor]): 输入张量
            attn_mask (Optional[torch.Tensor], optional): 注意力掩码矩阵 Defaults to None.
            key_padding_mask (Optional[torch.Tensor], optional): 词嵌入掩码矩阵 Defaults to None.

        Returns:
            Optional[torch.Tensor]: 输出张量
        """
        if self.norm_first:
            x = x + self.drop1(
                self.self_attn(
                    query=self.norm1(x),
                    key=None,
                    value=None,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self.drop1(
                    self.self_attn(
                        query=x,
                        key=None,
                        value=None,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                    )
                )
            )
            x = self.norm2(x + self.drop2(self.ffn(x)))
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
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        """
        Transformer Decoder初始化

        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            d_ff (int): 前馈神经网络维度
            dropout (float, optional): 神经元随机失活概率 Defaults to 0.1.
            norm_first (bool, optional): 是否先归一化 Defaults to False.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn = MultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Optional[torch.Tensor],
        memory: Optional[torch.Tensor],
        tgt_attn_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Transformer Decoder前向传播

        Args:
            x (Optional[torch.Tensor]): 输入张量
            memory (Optional[torch.Tensor]): Encoder输出
            tgt_attn_mask (Optional[torch.Tensor]): 解码器注意力掩码矩阵
            tgt_key_padding_mask (Optional[torch.Tensor]): 解码器词嵌入掩码矩阵
            memory_key_padding_mask (Optional[torch.Tensor]): 编码器词嵌入掩码矩阵

        Returns:
            Optional[torch.Tensor]: 输出张量
        """
        if self.norm_first:
            x = x + self.drop1(
                self.self_attn(
                    query=self.norm1(x),
                    key=None,
                    value=None,
                    attn_mask=tgt_attn_mask,
                    key_padding_mask=tgt_key_padding_mask,
                )
            )
            x = x + self.drop2(
                self.cross_attn(
                    query=self.norm2(x),
                    key=memory,
                    value=memory,
                    attn_mask=None,
                    key_padding_mask=memory_key_padding_mask,
                )
            )
            x = x + self.drop3(self.ffn(self.norm3(x)))
        else:
            x = self.norm1(
                x
                + self.drop1(
                    self.self_attn(
                        query=x,
                        key=None,
                        value=None,
                        attn_mask=tgt_attn_mask,
                        key_padding_mask=tgt_key_padding_mask,
                    )
                )
            )
            x = self.norm2(
                x
                + self.drop2(
                    self.cross_attn(
                        query=x,
                        key=memory,
                        value=memory,
                        attn_mask=None,
                        key_padding_mask=memory_key_padding_mask,
                    )
                )
            )
            x = self.norm3(x + self.drop3(self.ffn(x)))
        return x


@dataclass
class TransformerConfig:
    """
    Transformer配置类
    """

    d_model: int = 512  # 模型维度
    num_heads: int = 8  # 注意力头数
    d_ff: int = 2048  # 前馈神经网络维度
    num_layers: int = 6  # 编码器解码器层数
    dropout: float = 0.1  # 神经元随机失活概率

    norm_first: bool = False  # 是否先归一化
    pos_type: str = "sinusoidal"  # 位置编码方式
    seq_max_len: int = 512  # 处理序列最大长度


def _build_position(pos_type: str, seq_max_len: int, d_model: int):
    """
    获取指定类型的位置编码其
    Args:
        pos_type (str): 编码方式
        seq_max_len (int): 最大序列长度
        d_model (int): 模型维度

    Raises:
        ValueError: 无法识别的编码类型

    Returns:
        _type_: 编码器
    """
    if pos_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model=d_model, seq_max_len=seq_max_len)
    elif pos_type == "none":
        return None
    else:
        raise ValueError(f"Unknown positional encoding({pos_type}).")


class EncoderBone(nn.Module):
    """
    Transformer Encoder骨架
    """

    def __init__(self, config: TransformerConfig):
        """
        Transformer Encoder骨架初始化

        Args:
            config (TransformerConfig): Transformer配置信息
        """
        super(EncoderBone, self).__init__()

        self.config = config
        self.pos = _build_position(
            pos_type=config.pos_type,
            seq_max_len=config.seq_max_len,
            d_model=config.d_model,
        )
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
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
        Transformer Encoder前向传播

        Args:
            x (torch.Tensor): 输入张量
            key_padding_mask (Optional[torch.Tensor]): 词嵌入掩码矩阵

        Returns:
            Optional[torch.Tensor]: 输出张量
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
    Transformer Decoder骨架
    """

    def __init__(self, config: TransformerConfig):
        """
        Transformer Decoder骨架初始化

        Args:
            config (TransformerConfig): Transformer配置信息
        """
        super(DecoderBone, self).__init__()

        self.config = config
        self.pos = _build_position(
            pos_type=config.pos_type,
            seq_max_len=config.seq_max_len,
            d_model=config.d_model,
        )
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
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
    ):
        """
        Transformer Decoder前向传播

        Args:
            x (torch.Tensor): 输入张量
            memory (torch.Tensor): 编码器输出张量
            tgt_attn_mask (Optional[torch.Tensor]): 解码器注意力掩码矩阵
            tgt_key_padding_mask (Optional[torch.Tensor]): 解码器词嵌入掩码矩阵
            memory_key_padding_mask (Optional[torch.Tensor]): 编码器词嵌入掩码矩阵

        Returns:
            _type_: _description_
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


class OriginalTransformer(nn.Module):
    """
    原始Transformer类
    """

    def __init__(
        self,
        config: TransformerConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        tie_embeddings: bool = False,
    ):
        """
        Transformer初始化

        Args:
            config (TransformerConfig): Transformer配置类
            src_vocab_size (int): 源词表大小
            tgt_vocab_size (int): 目标词表大小
            tie_embeddings (bool, optional): 是否共享权重 Defaults to False.
        """
        super(OriginalTransformer, self).__init__()

        self.src_token_embed = TokenEmbedding(
            vocab_size=src_vocab_size, d_model=config.d_model
        )
        self.tgt_token_embed = TokenEmbedding(
            vocab_size=tgt_vocab_size, d_model=config.d_model
        )

        self.encode = EncoderBone(config=config)
        self.decode = DecoderBone(config=config)

        self.generator = nn.Linear(
            in_features=config.d_model, out_features=tgt_vocab_size, bias=False
        )

        if tie_embeddings:  # 共享权重
            self.generator.weight = self.tgt_token_embed.embed.weight

    def forward(
        self,
        src_token_ids: torch.Tensor,
        tgt_token_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_causal_mask: Optional[torch.Tensor] = None,
    ):
        """
        Transformer前向传播

        Args:
            src_token_ids (torch.Tensor): 源序列
            tgt_token_ids (torch.Tensor): 目标序列
            src_key_padding_mask (Optional[torch.Tensor], optional): 源词嵌入掩码 Defaults to None.
            tgt_key_padding_mask (Optional[torch.Tensor], optional): 目标词嵌入掩码 Defaults to None.
            tgt_causal_mask (Optional[torch.Tensor], optional): 因果注意力掩码 Defaults to None.

        Returns:
            _type_: 输出张量
        """
        _, len_t = tgt_token_ids.shape
        if tgt_causal_mask is None:
            tgt_causal_mask = make_causal_mask(len_t, src_token_ids.device)

        # 1. embedding
        src_embed = self.src_token_embed(src_token_ids)
        tgt_embed = self.tgt_token_embed(tgt_token_ids)

        # 2. encode
        memory = self.encode(x=src_embed, key_padding_mask=src_key_padding_mask)

        # 3. decode
        out = self.decode(
            x=tgt_embed,
            memory=memory,
            tgt_attn_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 4. generator
        logits = self.generator(out)

        return logits


def make_causal_mask(l: int, device: torch.device) -> torch.Tensor:
    """
    构建因果注意力掩码

    Args:
        l (int): 掩码大小
        device (torch.device): 设备

    Returns:
        torch.Tensor: 因果注意力掩码矩阵
    """
    return torch.triu(torch.ones(size=(l, l), device=device), diagonal=1).bool()


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
        dropout=0.1,
        norm_first=False,
        pos_type="sinusoidal",
        seq_max_len=512,
    )

    model = OriginalTransformer(
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
