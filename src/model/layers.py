import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_p: float = 0.0, resid_p: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_p)
        self.resid_drop = nn.Dropout(resid_p)

        # will re-create per sequence length
        self.register_buffer("mask", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, T, C)
        """
        B, T, C = x.shape

        # project and split heads
        qkv = self.qkv(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, nH, T, dH)

        # scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nH, T, T)
        if (self.mask is None) or (self.mask.size(-1) != T):
            self.mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nH, T, dH)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, attn_p=p, resid_p=p)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
