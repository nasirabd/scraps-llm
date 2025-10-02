import math, yaml, json, csv, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.tokenization.tokenizer import BPETok
from src.data.dataset import RecipesJSONL
from src.data.collate import pad_collate
from src.model.transformer import RecipeTransformer

from rouge_score import rouge_scorer
import sacrebleu, random


# -------------------------
# Helpers
# -------------------------
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, ckpt_dir=Path("model/checkpoints")):
        self.patience = patience
        self.delta = delta
        self.ckpt_dir = ckpt_dir
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, current_val_loss, model):
        improved = current_val_loss < (self.best - self.delta)
        if improved:
            self.best = current_val_loss
            self.bad_epochs = 0
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, self.ckpt_dir / "best_model.pt")
        else:
            self.bad_epochs += 1
        return improved, (self.bad_epochs >= self.patience)


def init_csv_logger(path: Path):
    new_file = not path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["epoch", "train_loss", "val_loss", "val_ppl", "lr", "epoch_time_sec"])
    return f, writer


@torch.no_grad()
def quick_generate(model, tok, device, ing="chicken, garlic, onion", max_new_tokens=80):
    prompt = f"Ingredients: { ing }\nRecipe:"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tok.eos_id:
            break
    return tok.decode(ids[0].tolist())
# -------------------------
# BLEU and ROUGE
# -------------------------
@torch.no_grad()
def _sample_continuation_for_eval(model, tok, device, prompt: str,
                                  max_new_tokens=160, temperature=0.7, top_k=50, top_p=0.9):
    """Lightweight sampler that returns ONLY the continuation (no prompt)."""
    # Encode prompt without EOS; add BOS only
    prompt_ids = tok.encode(prompt, add_special=False)
    ids = torch.tensor([[tok.bos_id] + prompt_ids], dtype=torch.long, device=device)
    prompt_len = ids.size(1)

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]                      # (1, V)
        # temperature
        logits = logits / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)

        # top-k
        if top_k and top_k > 0:
            _, topk_idx = torch.topk(probs, k=top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter(1, topk_idx, 1.0)
            probs = (probs * mask) / probs.sum(dim=-1, keepdim=True)

        # top-p (nucleus)
        if top_p and (0.0 < top_p < 1.0):
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            filtered = torch.zeros_like(probs).scatter(1, sorted_idx, keep.float() * sorted_probs)
            probs = filtered / filtered.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tok.eos_id:
            break

    all_ids = ids[0].tolist()
    cont_ids = all_ids[prompt_len:]  # strictly after prompt
    hyp = tok.decode(cont_ids)
    # TEMP readability clean for byte-level artifacts (remove if your tokenizer is fixed)
    hyp = hyp.replace("Ġ", " ").strip()
    return hyp

@torch.no_grad()
def eval_text_metrics(model, tok, device, rows, *,
                      max_eval=64, max_new_tokens=160, temperature=0.7, top_k=50, top_p=0.9):
    """
    rows: list[{'ingredients': str, 'recipe': str}]
    Returns: {'rougeL_f1': float, 'bleu': float}
    """
    # Sample a small subset for speed
    sample_rows = rows if len(rows) <= max_eval else random.sample(rows, max_eval)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs, hyps = [], []
    rouge_f1s = []

    for r in sample_rows:
        ing, ref = r["ingredients"], r["recipe"]
        prompt = f"Ingredients: {ing}\nRecipe:"
        hyp = _sample_continuation_for_eval(
            model, tok, device, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        refs.append(ref)
        hyps.append(hyp)
        rouge = scorer.score(ref, hyp)["rougeL"].fmeasure
        rouge_f1s.append(rouge)

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0
    rougeL_f1 = 100.0 * (sum(rouge_f1s) / max(1, len(rouge_f1s)))
    return {"rougeL_f1": rougeL_f1, "bleu": bleu}


# -------------------------
# Custom cosine scheduler
# -------------------------
def cosine_schedule(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# -------------------------
# Training
# -------------------------
def train(cfg):
    # normalize cfg types
    cfg["optim"]["lr"] = float(cfg["optim"]["lr"])
    cfg["optim"]["weight_decay"] = float(cfg["optim"]["weight_decay"])
    cfg["optim"]["grad_clip"] = float(cfg["optim"]["grad_clip"])

    device = "cuda" if torch.cuda.is_available() and cfg["device"] == "auto" else cfg["device"]
    torch.manual_seed(cfg["seed"])

    # --- Data ---
    tok = BPETok(cfg["vocab_path"])
    train_ds = RecipesJSONL("train", tok=tok, max_len=cfg["data"]["max_len"])
    val_ds   = RecipesJSONL("val",   tok=tok, max_len=cfg["data"]["max_len"])
    val_rows_cached = val_ds.rows

    collate = lambda b: pad_collate(b, tok.pad_id)
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["train_batch_size"],
                          shuffle=True, num_workers=cfg["data"]["num_workers"],
                          collate_fn=collate, pin_memory=torch.cuda.is_available())
    val_dl   = DataLoader(val_ds, batch_size=cfg["data"]["val_batch_size"],
                          shuffle=False, num_workers=cfg["data"]["num_workers"],
                          collate_fn=collate, pin_memory=torch.cuda.is_available())

    # --- Model ---
    model = RecipeTransformer(
        vocab_size=tok.vocab_size(),
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_len=cfg["model"]["max_len"],
        p=cfg["model"]["dropout"],
        tie_weights=cfg["model"]["tie_weights"],
    ).to(device)

    # --- Optimizer / AMP / Loss ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["optim"]["mixed_precision"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # --- Logging (TB + CSV) ---
    run_name = time.strftime("scraps-%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"TensorBoard logging to {run_dir}")

    ckpt_dir = Path("model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_f, csv_writer = init_csv_logger(ckpt_dir / "training_log.csv")

    early = EarlyStopping(patience=3, delta=0.0, ckpt_dir=ckpt_dir)

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * cfg["optim"]["epochs"]
    warmup_steps = cfg["optim"]["warmup_steps"]
    base_lr = cfg["optim"]["lr"]

    global_step = 0
    for epoch in range(cfg["optim"]["epochs"]):
        epoch_start = time.time()
        model.train()
        running = 0.0

        for x, y in train_dl:
            global_step += 1
            # update LR manually
            lr = cosine_schedule(global_step, warmup_steps, total_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast(enabled=cfg["optim"]["mixed_precision"]):
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.reshape(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            scaler.step(opt)
            scaler.update()

            running += loss.item()

        avg_train = running / len(train_dl)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=cfg["optim"]["mixed_precision"]):
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.reshape(-1))
                val_loss += loss.item()
        avg_val = val_loss / max(1, len(val_dl))
        val_ppl = math.exp(min(20, avg_val))
        epoch_time = time.time() - epoch_start

        # Print
        print(f"epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}, ppl={val_ppl:.2f}, lr={lr:.6g}, t={epoch_time:.1f}s")
        

        # TensorBoard
        writer.add_scalar("loss/train", avg_train, epoch + 1)
        writer.add_scalar("loss/val", avg_val, epoch + 1)
        writer.add_scalar("metrics/perplexity_val", val_ppl, epoch + 1)
        writer.add_scalar("lr", lr, epoch + 1)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch + 1)
        writer.add_text("samples/chicken_garlic_onion", quick_generate(model, tok, device), epoch + 1)
        writer.flush()

        # CSV
        csv_writer.writerow([epoch + 1, f"{avg_train:.6f}", f"{avg_val:.6f}", f"{val_ppl:.6f}", f"{lr:.8f}", f"{epoch_time:.2f}"])
        csv_f.flush()

        # Save "last"
        torch.save({"model": model.state_dict()}, ckpt_dir / "last_model.pt")

        # Early stopping
        improved, stop = early.step(avg_val, model)

        # ----- Optional: in-loop text eval (ROUGE-L F1 + BLEU) -----
        evcfg = cfg.get("eval", {})
        if evcfg.get("in_loop", True) and ((epoch + 1) % evcfg.get("every", 2) == 0):
            m = eval_text_metrics(
                model, tok, device, rows=val_rows_cached,
                max_eval=evcfg.get("max_examples", 64),
                max_new_tokens=evcfg.get("max_new_tokens", 160),
                temperature=evcfg.get("temperature", 0.7),
                top_k=evcfg.get("top_k", 50),
                top_p=evcfg.get("top_p", 0.9),
            )
            print(f"metrics: ROUGE-L F1={m['rougeL_f1']:.2f}  BLEU={m['bleu']:.2f}")
            writer.add_scalar("text/rougeL_f1", m["rougeL_f1"], epoch + 1)
            writer.add_scalar("text/bleu",       m["bleu"],       epoch + 1)
            writer.flush()

        if stop:
            print(f"Early stopping at epoch {epoch+1} (best val_loss={early.best:.4f}).")
            break

    writer.close()
    csv_f.close()


if __name__ == "__main__":
    with open("configs/train_small.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
