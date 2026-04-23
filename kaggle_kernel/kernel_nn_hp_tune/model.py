"""FT-Transformer with digit + numeric + categorical tokens.

Key difference vs prior FT-Transformer null: digit columns get their
own 10-way embedding tables (one per column x decimal position), so
the NN can see the quantisation signal that drove digit-XGB's lift.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class NumericalTokenizer(nn.Module):
    """x (B, N) -> (B, N, d). Per-feature learnable weight+bias."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class CategoricalTokenizer(nn.Module):
    """Per-column embedding tables. Works for cats and digit columns."""
    def __init__(self, cardinalities: list[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=math.sqrt(5))

    def forward(self, x):  # (B, C) long -> (B, C, d)
        return torch.stack(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )


class CLSToken(nn.Module):
    def __init__(self, d_token: int):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.token, a=math.sqrt(5))

    def forward(self, x):
        return torch.cat([self.token.expand(x.size(0), 1, -1), x], dim=1)


class FTBlock(nn.Module):
    def __init__(self, d_token, n_heads, attn_drop, ffn_drop, ffn_factor):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=attn_drop, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_token)
        d_hidden = int(d_token * ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(d_hidden, d_token),
        )

    def forward(self, x):
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.norm_ffn(x)
        return x + self.ffn(h)


class DigitFTTransformer(nn.Module):
    """Tokens: numerics, digit columns (embedded), cats, CLS. CLS -> 3 logits."""
    def __init__(self, n_num: int, digit_cards: list[int], cat_cards: list[int],
                 d_token: int = 128, n_blocks: int = 4, n_heads: int = 8,
                 attn_drop: float = 0.15, ffn_drop: float = 0.15,
                 ffn_factor: float = 4 / 3, n_classes: int = 3):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        self.num_tok = NumericalTokenizer(n_num, d_token)
        self.dig_tok = CategoricalTokenizer(digit_cards, d_token) if digit_cards else None
        self.cat_tok = CategoricalTokenizer(cat_cards, d_token) if cat_cards else None
        self.cls = CLSToken(d_token)
        self.blocks = nn.ModuleList([
            FTBlock(d_token, n_heads, attn_drop, ffn_drop, ffn_factor)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x_num, x_dig, x_cat):
        toks = [self.num_tok(x_num)]
        if self.dig_tok is not None and x_dig is not None and x_dig.size(1) > 0:
            toks.append(self.dig_tok(x_dig))
        if self.cat_tok is not None and x_cat is not None and x_cat.size(1) > 0:
            toks.append(self.cat_tok(x_cat))
        x = self.cls(torch.cat(toks, dim=1))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.head_norm(x[:, 0]))
