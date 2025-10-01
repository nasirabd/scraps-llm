# src/infer/generate.py
import argparse
from pathlib import Path
import yaml
import torch

from src.tokenization.tokenizer import BPETok
from src.model.transformer import RecipeTransformer


# -----------------------------
# Sampling helpers
# -----------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    probs = torch.softmax(logits, dim=-1)

    # Top-k
    if top_k and top_k > 0:
        _, topk_idx = torch.topk(probs, k=top_k)
        mask = torch.zeros_like(probs)
        mask.scatter_(0, topk_idx, 1.0)
        probs = probs * mask
        probs = probs / probs.sum()

    # Top-p (nucleus)
    if top_p and (0.0 < top_p < 1.0):
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True  # always keep top-1
        mask = torch.zeros_like(probs)
        mask.scatter_(0, sorted_idx[keep], 1.0)
        probs = probs * mask
        probs = probs / probs.sum()

    return probs


def _clean_readability(s: str) -> str:
    # Temporary readability pass for byte-level artifacts.
    s = s.replace("Ġ", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


@torch.no_grad()
def generate(
    model,
    tok,
    prompt_text: str,
    device: str = "cpu",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_at_eos: bool = True,
):
    """
    Token-level continuation:
      - encodes prompt WITHOUT EOS, prepends BOS only
      - returns (full_text, continuation_only)
    """
    model.eval()

    prompt_ids = tok.encode(prompt_text, add_special=False)     # no BOS/EOS from tokenizer
    ids = torch.tensor([[tok.bos_id] + prompt_ids], dtype=torch.long, device=device)
    prompt_len = ids.size(1)  # for token-level slicing

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]             # (1, V)
        logits = logits.squeeze(0) / max(1e-6, temperature)
        probs = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        next_id = torch.multinomial(probs, num_samples=1)       # (1,)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if stop_at_eos and next_id.item() == tok.eos_id:
            break

    all_ids = ids[0].tolist()
    cont_ids = all_ids[prompt_len:]  # strictly new tokens after the prompt

    full_text = tok.decode(all_ids)
    continuation = tok.decode(cont_ids)
    continuation = _clean_readability(continuation)
    return full_text, continuation


# -----------------------------
# Load model + ckpt
# -----------------------------
def load_from_config(config_path: str, ckpt_dir="model/checkpoints", vocab_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tok_path = vocab_path or cfg.get("vocab_path", "tokenizer/bpe.json")
    tok = BPETok(tok_path)

    mcfg = cfg["model"]
    model = RecipeTransformer(
        vocab_size=tok.vocab_size(),
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        max_len=mcfg["max_len"],
        p=mcfg.get("dropout", 0.0),
        tie_weights=mcfg.get("tie_weights", True),
    ).to(device)

    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "last_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir} (best_model.pt or last_model.pt)")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, tok, device


# -----------------------------
# Prompt scaffold
# -----------------------------
def build_prompt(ingredients: str, scaffold: str) -> str:
    """Keep what the model sees consistent with training, but allow helpful scaffolds."""
    if scaffold == "basic":
        return f"Ingredients: {ingredients}\nRecipe:"
    elif scaffold == "title_step1":
        return f"Ingredients: {ingredients}\nRecipe:\nTitle:\nStep 1:"
    # fallback
    return f"Ingredients: {ingredients}\nRecipe:"


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate a recipe from ingredients with Scraps-LLM")
    p.add_argument("--config", type=str, default="configs/train_small.yaml", help="YAML used to build the model")
    p.add_argument("--vocab",  type=str, default=None, help="Optional override for tokenizer path")
    p.add_argument("--ckpt_dir", type=str, default="model/checkpoints", help="Folder with best_model.pt/last_model.pt")

    p.add_argument("--ingredients", type=str, default="chicken, garlic, onion", help="Comma-separated ingredient list")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--no_eos_stop", action="store_true", help="Do not stop on EOS")

    # UX flags
    p.add_argument("--just_recipe", action="store_true", default=True, help="Print only the generated recipe")
    p.add_argument("--show_full", action="store_true", help="Also print full decoded text (debug)")
    p.add_argument("--scaffold", type=str, default="basic",
                   choices=["basic", "title_step1"],
                   help="Prompt scaffold style")
    p.add_argument("--num_recipes", type=int, default=1, help="Number of recipes to generate")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    model, tok, device = load_from_config(args.config, ckpt_dir=args.ckpt_dir, vocab_path=args.vocab, device=None)

    prompt = build_prompt(args.ingredients, args.scaffold)

    for i in range(args.num_recipes):
        full_text, continuation = generate(
            model, tok, prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_at_eos=(not args.no_eos_stop),
        )

        print(f"\n=== Recipe {i+1} ===")
        if args.show_full:
            print("\n--- Full (prompt + recipe) ---")
            print(full_text)

        if args.just_recipe:
            print("\n--- Generated Recipe ---")
            print(continuation)
        else:
            print("\n--- Ingredients ---")
            print(args.ingredients)
            print("\n--- Generated Recipe ---")
            print(continuation)


if __name__ == "__main__":
    main()
