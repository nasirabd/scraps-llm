from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.normalizers import NFKC, Lowercase, Sequence

# Fallback reader if import path changes
try:
    from src.preprocess import ingest_jsonl  # must yield dicts
except Exception:
    def ingest_jsonl(p: Path):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<ING>", "<REC>", "<TITLE>", "<STEP>"]

def parse_args():
    ap = argparse.ArgumentParser("Build a ByteLevel BPE tokenizer for Scraps-LLM")
    ap.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", nargs="+", default=["train"], help="e.g. train val test")
    ap.add_argument("--fields", nargs="+", default=["ingredients", "recipe", "title"])
    ap.add_argument("--out_dir", type=Path, default=Path("tokenizer"))
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--nfkc", action="store_true")
    ap.add_argument("--sample_limit", type=int, default=0, help="0 = all")
    return ap.parse_args()

def iter_corpus(data_dir: Path, splits: List[str], fields: List[str], sample_limit: int) -> Iterable[str]:
    """Yield compact, structured strings so the tokenizer learns tags in context."""
    count = 0
    for split in splits:
        p = data_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        for obj in ingest_jsonl(p):
            parts = []
            if "title" in fields and obj.get("recipe", ""):
                # many recipes begin with "Title: ..."
                title_line = (obj["recipe"].split("\n", 1)[0] or "").strip()
                if title_line:
                    parts.append(f"<TITLE> {title_line}")
            if "ingredients" in fields and obj.get("ingredients"):
                parts.append(f"<ING> {obj['ingredients']}")
            if "recipe" in fields and obj.get("recipe"):
                parts.append(f"<REC> {obj['recipe']}")
            if parts:
                yield " ".join(parts)
                count += 1
                if sample_limit and count >= sample_limit:
                    return

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalization
    norms = []
    if args.nfkc: norms.append(NFKC())
    if args.lower: norms.append(Lowercase())
    if norms:
        tok.normalizer = Sequence(norms)

    # ByteLevel BPE setup
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIALS,
        show_progress=True,
    )

    print(
        f"Training BPE on {args.data_dir} splits={args.splits} fields={args.fields} "
        f"→ vocab={args.vocab_size} min_freq={args.min_frequency} lower={args.lower} nfkc={args.nfkc}"
    )
    tok.train_from_iterator(
        iter_corpus(args.data_dir, args.splits, args.fields, args.sample_limit),
        trainer=trainer
    )

    # Save tokenizer + sidecar meta
    tok_path = args.out_dir / "bpe.json"
    tok.save(str(tok_path))

    vocab = tok.get_vocab()
    meta = {
        "vocab_size": len(vocab),
        "specials": {t: vocab.get(t, -1) for t in SPECIALS},
        "lower": bool(args.lower),
        "nfkc": bool(args.nfkc),
    }
    with (args.out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Quick sanity
    test = "[BOS] <ING> butter, sugar <REC> Title: Test\nStep 1: mix. [EOS]"
    print("Sample encode (first 16 ids):", tok.encode(test).ids[:16])
    print(f"✅ Saved tokenizer → {tok_path}")
    print(f"🧾 Meta → {args.out_dir/'meta.json'}")

if __name__ == "__main__":
    main()
