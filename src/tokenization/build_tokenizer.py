from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor

from src.preprocess import ingest_jsonl

DATA = Path("data/processed")
TOKDIR = Path("tokenizer")
TOKDIR.mkdir(parents=True, exist_ok=True)

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<ING>", "<REC>", "<TITLE>", "<STEP>"]

def iter_corpus():
    """Yield only the text fields (not raw JSON lines)."""
    for split in ("train", "val", "test"):
        p = DATA / f"{split}.jsonl"
        if not p.exists():
            continue
        for obj in ingest_jsonl(p):
            ing = obj.get("ingredients","")
            rec = obj.get("recipe","")
            if ing: 
                yield ing
            if rec: 
                yield rec

if __name__ == "__main__":
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel()
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=16000,
        min_frequency=2,
        special_tokens=SPECIALS,
        show_progress=True,
    )

    tok.train_from_iterator(iter_corpus(), trainer=trainer)

    out = TOKDIR / "bpe.json"
    tok.save(str(out))
    print(f"✅ Saved tokenizer to {out}")