from pathlib import Path
import json, torch
from src.tokenization.bpe_tok import BPETok  # your BPETok

TOK = "tokenizer/bpe.json"
IN_DIR = Path("data/processed")
OUT_DIR = Path("data/cache")
MAX_LEN = 256

def build_split(split: str):
    inp = IN_DIR / f"{split}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outp = OUT_DIR / f"{split}.pt"
    tok = BPETok(TOK)

    ids_list = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            text = f"Ingredients: {o['ingredients']}\nRecipe: {o['recipe']}"
            enc = tok.encode(text, add_special=True, max_len=MAX_LEN)
            ids = enc.input_ids if hasattr(enc, "input_ids") else (enc.ids if hasattr(enc, "ids") else enc)
            if not ids:
                continue
            ids_list.append(torch.tensor(ids, dtype=torch.long))

    torch.save({"ids": ids_list, "pad_id": tok.pad_id}, outp)
    print(f"✓ wrote {outp}  ({len(ids_list)} sequences)")

if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        if (IN_DIR / f"{s}.jsonl").exists():
            build_split(s)
