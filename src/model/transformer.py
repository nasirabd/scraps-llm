# transformer.py
import torch
import torch.nn as nn
from .layers import Block, RMSNorm

class RecipeTransformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style) with:
      - RMSNorm
      - SwiGLU MLP
      - RoPE (rotary) on Q/K inside attention
      - SDPA when available, masked-matmul fallback otherwise
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_len: int = 1024,
        p: float = 0.1,
        use_rope: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.max_len = max_len
        self.use_rope = use_rope

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Absolute positional embeddings only if NOT using RoPE
        if not use_rope:
            self.pos_emb = nn.Embedding(max_len, d_model)
        else:
            self.register_parameter("pos_emb", None)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, p=p, use_rope=use_rope) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        if T > self.max_len:
            idx = idx[:, -self.max_len:]
            T = idx.size(1)

        x = self.tok_emb(idx)  # (B,T,C)

        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        return self.head(x)  # (B,T,V)
