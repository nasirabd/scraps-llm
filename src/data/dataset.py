import json
from pathlib import Path
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from src.utils.io import read_jsonl

class RecipesJSONL(Dataset):
    def __init__(self, split="train", tok=None, max_len=512, keep_raw=False,
                 root="data/processed", cache_dir="data/cache"):
        self.split = split
        self.keep_raw = keep_raw
        self.max_len = max_len
        self.tok = tok
        self.cached = False

        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.root / f"{split}.jsonl"
        cache_p = self.cache_dir / f"{split}.pt"

        if cache_p.exists():
            print(f"Loading from cache: {cache_p.name}")
            # show progress while reading large .pt file
            with tqdm(total=1, desc=f"Reading {cache_p.name}", ncols=80) as pbar:
                blob = torch.load(cache_p, map_location="cpu")
                pbar.update(1)

            blob = torch.load(cache_p, map_location="cpu")
            self.ids_list = blob["ids"]
            self.pad_id = blob.get("pad_id", 0)
            self.cached = True
            return  

        if not self.path.exists():
            raise FileNotFoundError(f"Missing {self.path}. Run preprocessing or build cache.")
        if tok is None:
            raise ValueError("Tokenizer `tok` is required when cache is missing.")

        # Only load rows when not cached
        self.rows = list(read_jsonl(self.path))

    def __len__(self):
        return len(self.ids_list) if self.cached else len(self.rows)

    @staticmethod
    def _to_text(ingredients: str, recipe: str) -> str:
        return f"Ingredients: {ingredients}\nRecipe: {recipe}"

    def __getitem__(self, idx: int):
        if self.cached:
            ids = self.ids_list[idx].tolist()
            if len(ids) < 2:
                ids = ids + [getattr(self, "pad_id", 0)]
            if self.keep_raw:
                return {"ids": ids, "text": None, "ingredients": None, "recipe": None}
            return ids

        r = self.rows[idx]
        ing = r.get("ingredients", "")
        rec = r.get("recipe", "")
        text = self._to_text(ing, rec)

        enc = self.tok.encode(text, add_special=True, max_len=self.max_len)
        # normalize to list[int]
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
