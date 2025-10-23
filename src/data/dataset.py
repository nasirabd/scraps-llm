import json
from pathlib import Path
from typing import Optional, List

from torch.utils.data import Dataset
from src.utils.io import read_jsonl

class RecipesJSONL(Dataset):
    """
    Loads processed JSONL with pairs like:
      {"ingredients": "...", "recipe": "Title: ...\\nStep 1: ..."}
    and returns token id sequences using a tokenizer wrapper (BPETok).

    Args:
        split: "train" | "val" | "test"
        tok: tokenizer wrapper (must have .encode())
        max_len: truncate sequences to this length (after adding BOS/EOS)
        keep_raw: if True, also returns raw text alongside ids (debugging)
    """
    def __init__(
        self, 
        split: str = "train", 
        tok=None, 
        max_len: int = 512, 
        keep_raw: bool = False, 
        root: str = "data/processed"
    ):
        self.path = Path(root) / f"{split}.jsonl"
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing {self.path}. Run preprocessing first (e.g., "
                f"`make preprocess`."
            )
        self.rows = list(read_jsonl(self.path))
        if tok is None:
            raise ValueError("Tokenizer `tok` is required (e.g., BPETok('tokenizer/bpe.json')).")
        self.tok = tok
        self.max_len = max_len
        self.keep_raw = keep_raw

    def __len__(self):
        return len(self.rows)
    
    @staticmethod
    def _to_text(ingredients: str, recipe: str) -> str:
        return f"Ingredients: {ingredients}\nRecipe: {recipe}"

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        text = self._to_text(r["ingredients"], r["recipe"])

        ids = self.tok.encode(text, add_special=True)
        # --- SAFETY: normalize to list[int] ---
        if hasattr(ids, "input_ids"):      
            ids = list(ids.input_ids)
        elif hasattr(ids, "ids"):           
            ids = list(ids.ids)
        else:
            ids = list(ids)                
        if self.max_len is not None and len(ids) > self.max_len:
            ids = ids[:self.max_len]
        if self.keep_raw:
            return {"ids": ids, "text": text, "ingredients": r["ingredients"], "recipe": r["recipe"]}
        return ids
