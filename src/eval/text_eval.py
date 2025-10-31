import argparse, json, random
from pathlib import Path

import torch
from torch.cuda.amp import autocast

import yaml
import sacrebleu
from rouge_score import rouge_scorer

from src.tokenization.tokenizer import BPETok
from src.model.transformer import RecipeTransformer


@torch.inference_mode()
def sample_continuation(model, tok, device, prompt: str,
                        max_new_tokens=160, temperature=0.7, top_k=50, top_p=0.9):
    """Lightweight sampler (BOS + prompt, no EOS in prompt). Returns decoded continuation only."""
    # encode prompt without specials; add BOS only
    enc = tok.encode(prompt, add_special=False)
    if hasattr(enc, "input_ids"):
        prompt_ids = list(enc.input_ids)
    elif hasattr(enc, "ids"):
        prompt_ids = list(enc.ids)
    else:
        prompt_ids = list(enc)  # already a list

    ids = torch.tensor([[tok.bos_id] + prompt_ids], dtype=torch.long, device=device)
    prompt_len = ids.size(1)  # track prompt length for continuation slicing
    for _ in range(max_new_tokens):
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(ids)[:, -1, :]  # (1, V)

        logits = logits.float()
        logits = logits / max(1e-6, temperature)

        # --- Top-k on logits (before softmax) ---
        if top_k and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, topk_idx, 0.0)
            logits = logits + mask  # keep top-k, -inf others

        # --- Top-p (nucleus) on logits ---
        if top_p and (0.0 < top_p < 1.0):
            sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            # set dropped logits to -inf
            drop = ~keep
            sorted_logits = sorted_logits.masked_fill(drop, float('-inf'))
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)

        # guard against tiny numerical issues
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tok.eos_id:
            break


    # token-level slice to continuation
    all_ids = ids[0].tolist()
    cont_ids = all_ids[prompt_len:]
    hyp = tok.decode(cont_ids)
    # temporary cleanup if your tokenizer shows "Ġ" artifacts
    hyp = hyp.replace("Ġ", " ").strip()
    return hyp


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tok(cfg, ckpt_dir="model/checkpoints", vocab_override=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETok(vocab_override or cfg.get("vocab_path", "tokenizer/bpe.json"))
    mcfg = cfg["model"]
    model = RecipeTransformer(
        vocab_size=tok.vocab_size(),
        d_model=mcfg["d_model"], n_layers=mcfg["n_layers"], n_heads=mcfg["n_heads"],
        max_len=mcfg["max_len"], p=mcfg.get("dropout", 0.0), tie_weights=mcfg.get("tie_weights", True)
    ).to(device)
    ckpt_dir = Path(ckpt_dir)
    ckpt = ckpt_dir / "best_model.pt"
    if not ckpt.exists():
        ckpt = ckpt_dir / "last_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir} (best_model.pt/last_model.pt)")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, tok, device


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate text metrics (ROUGE-L F1, BLEU, coverage) on a split.")
    ap.add_argument("--config", default="configs/train_small.yaml")
    ap.add_argument("--ckpt_dir", default="model/checkpoints")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=200, help="evaluate first N examples (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle before limiting")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--save_csv", default="", help="optional: path to save predictions CSV")
    ap.add_argument("--out_dir", default="", help="optional: path to save predictions json")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    model, tok, device = load_model_and_tok(cfg, args.ckpt_dir, args.vocab)

    data_path = Path(cfg["data"]["processed_dir"]) / f"{args.split}.jsonl"
    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines()]
    if args.shuffle:
        random.Random(123).shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs, hyps = [], []
    out_rows = []

    for r in rows:
        ing = r["ingredients"]
        ref = r["recipe"]
        prompt = f"Ingredients: {ing}\nRecipe:"
        hyp = sample_continuation(
            model, tok, device, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        refs.append(ref); hyps.append(hyp)
        out_rows.append((ing, ref, hyp))

    # Metrics
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0
    rouge_f1s = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(refs, hyps)]
    rougeL_f1 = 100.0 * (sum(rouge_f1s) / max(1, len(rouge_f1s)))
    

    print(f"Split: {args.split}  N={len(hyps)}")
    print(f"ROUGE-L F1: {rougeL_f1:.2f}")
    print(f"BLEU:       {bleu:.2f}")
  

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ingredients", "reference", "hypothesis"])
            w.writerows(out_rows)
        print(f"Saved predictions to {args.save_csv}")

        # Also save predictions JSON alongside CSV
        json_path = Path(args.save_csv).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            preds = [
                {"ingredients": ing, "reference": ref, "hypothesis": hyp}
                for ing, ref, hyp in out_rows
            ]
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {json_path}")
    
    # Always save metrics.json
    outdir = Path(args.out_dir) / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"split": args.split, "N": len(hyps),
             "rougeL_f1": rougeL_f1, "bleu": bleu},
            f, ensure_ascii=False, indent=2
        )
    print(f"Saved metrics to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
