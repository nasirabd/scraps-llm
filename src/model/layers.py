
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- RMSNorm ----------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)

# ---------- SwiGLU MLP ----------
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 4.0, p: float = 0.0):
        super().__init__()
        inner = int(mult * d_model)
        self.w1 = nn.Linear(d_model, inner, bias=True)  # gate
        self.w2 = nn.Linear(d_model, inner, bias=True)  # value
        self.w3 = nn.Linear(inner, d_model, bias=True)  # proj back
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)   # SwiGLU
        return self.drop(self.w3(x))

# ---------- RoPE helpers ----------
def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B,H,T,D), cos/sin: (T,D)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos_t = cos[..., ::2]
    sin_t = sin[..., ::2]
    y1 = x1 * cos_t - x2 * sin_t
    y2 = x1 * sin_t + x2 * cos_t
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y

def build_rope_cache(T: int, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    # standard theta base
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device) / dim))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,d->td", t, inv_freq)      # (T, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)           # (T, dim)
    return emb.cos(), emb.sin()                       # (T, dim)

# ---------- SDPA fallback ----------
def _causal_mask(T: int, device):
    return torch.ones(T, T, device=device, dtype=torch.bool).triu(1)

def _attn_fwd(q, k, v, dropout_p: float, training: bool):
    """
    Use PyTorch SDPA if available (torch>=2.0), else do masked matmul-softmax.
    q,k,v: (B,H,T,D)
    """
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p if training else 0.0,
            is_causal=True
        )
    # fallback for torch 1.x (cu116 etc.)
    B, H, T, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,T)
    scores = scores.masked_fill(_causal_mask(T, q.device), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if training and dropout_p and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, v)  # (B,H,T,D)

# ---------- Attention ----------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_p: float = 0.0, resid_p: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(attn_p)
        self.resid_drop = nn.Dropout(resid_p)

        # RoPE buffers (lazy-built per seq length)
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)
        self._rope_len = 0

    def _maybe_build_rope(self, T: int, device):
        if not self.use_rope:
            return
        need = (self._rope_cos is None) or (self._rope_len < T) or (self._rope_cos.device != device)
        if need:
            cos, sin = build_rope_cache(T, self.head_dim, device)
            self._rope_cos, self._rope_sin = cos, sin
            self._rope_len = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            self._maybe_build_rope(T, x.device)
            cos = self._rope_cos[:T]
            sin = self._rope_sin[:T]
            q = _rope_apply(q, cos, sin)
            k = _rope_apply(k, cos, sin)

        attn = _attn_fwd(q, k, v, dropout_p=self.attn_drop.p, training=self.training)  # (B,H,T,D)
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

# ---------- Transformer Block ----------
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p: float = 0.0, use_rope: bool = True):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, attn_p=p, resid_p=p, use_rope=use_rope)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mult=4.0, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
