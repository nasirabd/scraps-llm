import torch
import torch.nn as nn
from .layers import Block

class RecipeTransformer(nn.Module):
    """
    Minimal decoder-only Transformer (GPT-style) for autoregressive LM.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_layers: int = 2,
        n_heads: int = 6,
        max_len: int = 512,
        p: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([Block(d_model=d_model, n_heads=n_heads, p=p) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # optionally tie output projection with input embeddings (common in GPTs)
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) LongTensor of token IDs
        returns: (B, T, V) logits over vocabulary
        """
        B, T = idx.shape
        if T > self.max_len:
            idx = idx[:, -self.max_len:]
            T = idx.size(1)

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)                 # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        return logits
