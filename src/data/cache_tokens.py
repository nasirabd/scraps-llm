"""
Pre-tokenize all processed JSONL splits into torch tensors for faster training.
Reads max_len automatically from a YAML config file.
"""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from src.tokenization.tokenizer import BPETok
from src.data.dataset import RecipesJSONL


def load_max_len(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg.get("data", {}).get("max_len", 512))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_small.yaml",
                    help="YAML config path (used to read data.max_len)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out_dir", type=str, default="data/cache")
    args = ap.parse_args()

    max_len = load_max_len(args.config)
    print(f"📏 Using max_len={max_len} from {args.config}")

    tok = BPETok("tokenizer/bpe.json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"⚙️  Caching {split} split …")
        ds = RecipesJSONL(split, tok=tok, max_len=max_len, keep_raw=False)
        encoded = []
        for i in tqdm(range(len(ds))):
            ids = ds[i]
            encoded.append(torch.tensor(ids, dtype=torch.long))
        blob = {"ids": encoded, "pad_id": tok.pad_id, "max_len": max_len}
        torch.save(blob, out_dir / f"{split}.pt")
        print(f"✅ Saved {len(encoded)} to {out_dir / f'{split}.pt'}")

