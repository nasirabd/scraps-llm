import torch
from src.model.transformer import RecipeTransformer

V = 16000  # or tok.vocab_size()
model = RecipeTransformer(vocab_size=V, d_model=128, n_layers=2, n_heads=4, max_len=128, p=0.1)
x = torch.randint(0, V, (2, 50))  # (batch=2, seq=50)
logits = model(x)
print(logits.shape)  # -> torch.Size([2, 50, 16000])
