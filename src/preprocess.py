import argparse, csv, random
from pathlib import Path 
from typing import Dict, Iterable, List, Tuple

#importing utility functions
from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import clean

print(">>> Preprocess starting...")
#Create parser arguments for modfiying command line
def parse_args():
    p = argparse.ArgumentParser(description="Scraps-LLM preprocessing")
    p.add_argument("--raw", type=Path, required=True, help="Path to raw file (jsonl or csv)")
    p.add_argument("--format", choices=["jsonl","csv"], required=True, help="Input format")
    p.add_argument("--outdir", type=Path, default=Path("data/processed"), help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Split seed")
    p.add_argument("--train", type=float, default=.9, help="Train ratio")
    p.add_argument("--val", type=float, default=.05, help="Val ratio")
    p.add_argument("--test", type=float, default=.05, help="Test ratio")
    p.add_argument("--ing-col", type=str, default="ingredients", help="CSV column name for ingredients")
    p.add_argument("--rec-col", type=str, default="recipe", help="CSV column name for recipe")
    p.add_argument("--lower-ingredients", action="store_true", help="lowercase ingredients text")
    return p.parse_args()

def ingest_jsonl(path: Path) -> Iterable[Dict]:
    for obj in read_jsonl(path):
        yield{
            "ingredients": clean(obj["ingredients"]),
            "recipe": clean(obj["recipe"]),
        }

def ingest_csv(path: Path, ing_col: str, rec_col: str) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = clean(row[ing_col])
            rec = clean(row[rec_col])
            yield{
                "ingredients":ing,
                "recipe": rec,
            }

def split(items: List[Dict], train: float, val: float, test: float, seed: int) -> Tuple[List[Dict]]:
    assert abs(train + val + test - 1.0) < 1e-6
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_tr = int(n * train)
    n_va = int(n * val)
    tr = items[:n_tr] or items
    va = items[n_tr:n_tr+n_va] or items[:1]
    te = items[n_tr+n_va:] or items[:1]
    return tr,va,te 

def to_prompt_style(rows: Iterable[Dict]) -> Iterable[Dict]:
    for r in rows:
        yield{
            "ingredients": r["ingredients"],
            "recipe": r["recipe"]
        }

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)

    #Ingest
    if args.format == "jsonl":
        rows = list(ingest_jsonl(args.raw))
    else:
        rows = list(ingest_csv(args.raw,arg.ing_col,args.rec_col))
    
    if args.lower_ingredients:
        for r in rows:
            r["ingredients"] = r["ingredients"].lower()
    
    rows = list(to_prompt_style(rows))

    #Split
    tr,va,te = split(rows, args.train, args.val, args.test, args.seed)
    # test output
    # print("Sample row before writing:", rows[0], type(rows[0]))

    #Write
    write_jsonl(args.outdir / "train.jsonl", tr)
    write_jsonl(args.outdir / "val.jsonl", va)
    write_jsonl(args.outdir / "test.jsonl", te)

    print("✅ Wrote:", *(p.name for p in (args.outdir / "train.jsonl", args.outdir / "val.jsonl", args.outdir / "test.jsonl")))

if __name__ == "__main__":
    main()
print(">>> Done, wrote train/val/test JSONL files")
