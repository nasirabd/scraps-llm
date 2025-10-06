from src.tokenization.tokenizer import BPETok

tok = BPETok("tokenizer/bpe.json")

# Encode text → token IDs
ids = tok.encode("Ingredients: chicken, garlic\nRecipe: Title: Garlic Chicken")
print("IDs:", ids[:20])

# Decode back → text
print("Text:", tok.decode(ids))


# Inspect special token IDs
print("PAD:", tok.pad_id, "BOS:", tok.bos_id, "EOS:", tok.eos_id, "UNK:", tok.unk_id)

# Vocab size
print("Vocab size:", tok.vocab_size())