import json
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from src.utils.io import read_jsonl

class RecipesJSONL(Dataset):
    def __init__(self, split="train", tok=None, max_len=512, keep_raw=False,
                 root="data/processed", cache_dir="data/cache"):
        self.keep_raw = keep_raw
        self.max_len = max_len
        self.tok = tok 
        self.cached = False

        cache_p = Path(cache_dir) / f"{split}.pt"
        if cache_p.exists():
            print("Loading from cache...")
            blob = torch.load(cache_p, map_location="cpu")
            self.ids_list = blob["ids"]     
            self.pad_id = blob.get("pad_id", 0)
            self.cached = True
            return

        self.path = Path(root) / f"{split}.jsonl"
        if not self.path.exists():
            raise FileNotFoundError(f"Missing {self.path}. Run preprocessing or build cache.")
        if tok is None:
            raise ValueError("Tokenizer `tok` is required when cache is missing.")
        self.rows = list(read_jsonl(self.path))

    def __len__(self):
        return len(self.ids_list) if self.cached else len(self.rows)

    @staticmethod
    def _to_text(ingredients: str, recipe: str) -> str:
        return f"Ingredients: {ingredients}\nRecipe: {recipe}"

    def __getitem__(self, idx: int):
        if getattr(self, "cached", False):
            ids = self.ids_list[idx].tolist()
            if len(ids) < 2 and hasattr(self, "pad_id"):
                ids = ids + [self.pad_id]
            if self.keep_raw:
                return {"ids": ids, "text": None, "ingredients": None, "recipe": None}
            return ids

        r = self.rows[idx]
        ing = r.get("ingredients", "")
        rec = r.get("recipe", "")
        text = self._to_text(ing, rec)

        enc = self.tok.encode(text, add_special=True, max_len=self.max_len)
        # normalize to a plain list[int] 
        if hasattr(enc, "input_ids"):
            ids = list(enc.input_ids)
        elif hasattr(enc, "ids"):
            ids = list(enc.ids)
        else:
            ids = list(enc)
        if len(ids) < 2:
            ids = ids + [self.tok.pad_id]
        if self.keep_raw:
            return {"ids": ids, "text": text, "ingredients": ing, "recipe": rec}
        return ids
