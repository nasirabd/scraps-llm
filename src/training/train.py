import math, yaml, json, csv, time
from pathlib import Path
import os, shutil
from tqdm.auto import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
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
# --- Checkpoint helpers ---
def save_checkpoint(path: Path, *, model, optimizer=None, scaler=None,
                    epoch: int = 0, step: int = 0, optimizer_step: int = 0,
                    cfg: dict | None = None):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "optimizer_step": optimizer_step,
        "cfg": cfg or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

def load_checkpoint(path: Path, *, model, optimizer=None, scaler=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("optimizer_step", 0)
# --- simple k=v overrides: "optim.lr=1e-3,model.d_model=256,model.n_heads=8" ---
def _cast_scalar(s: str):
    # try bool
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    # try int
    try:
        return int(s)
    except ValueError:
        pass
    # try float
    try:
        return float(s)
    except ValueError:
        pass
    # fallback string
    return s

def apply_overrides(cfg: dict, overrides: str | None):
    if not overrides:
        return cfg
    pairs = [p.strip() for p in overrides.split(",") if p.strip()]
    for pair in pairs:
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        val = _cast_scalar(val.strip())
        # walk nested dict by dots
        cur = cfg
        parts = key.strip().split(".")
        for k in parts[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[parts[-1]] = val
    return cfg

class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, ckpt_dir=Path("model/checkpoints")):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, current_val_loss, model=None):
        improved = current_val_loss < (self.best - self.delta)
        if improved:
            self.best = current_val_loss
            self.bad_epochs = 0
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
    enc = tok.encode(prompt, add_special=True)
    ids = torch.tensor([enc.input_ids], dtype=torch.long, device=device)
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
    prompt_ids = tok.encode(prompt, add_special=False).input_ids
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
torch.set_float32_matmul_precision("high")          
torch.backends.cuda.matmul.allow_tf32 = True 
drive_dir = os.environ.get("DRIVE_CKPT_DIR")       
def train(cfg):
    # normalize cfg types
    cfg["optim"]["lr"] = float(cfg["optim"]["lr"])
    cfg["optim"]["weight_decay"] = float(cfg["optim"]["weight_decay"])
    cfg["optim"]["grad_clip"] = float(cfg["optim"]["grad_clip"])

    device = "cuda" if torch.cuda.is_available() and cfg["device"] == "auto" else cfg["device"]
    print("=== CUDA check ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version (PyTorch build):", torch.version.cuda)
        print(f"✓ Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"✓ Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print("Chosen device:", device)
    print("==================")
    torch.manual_seed(cfg["seed"])

    # --- Data ---
    tok = BPETok(cfg["vocab_path"])
    train_ds = RecipesJSONL("train", tok=tok, max_len=cfg["data"]["max_len"])
    val_ds   = RecipesJSONL("val",   tok=tok, max_len=cfg["data"]["max_len"])
   
    try:
        val_rows_cached = val_ds.rows  
    except AttributeError:
        # load rows directly from processed jsonl for evaluation-only usage
        data_path = Path(cfg["data"]["processed_dir"]) / "val.jsonl"
        with data_path.open("r", encoding="utf-8") as f:
            val_rows_cached = [json.loads(l) for l in f]


    collate = lambda b: pad_collate(b, tok.pad_id)
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["train_batch_size"],
                          shuffle=True, num_workers=cfg["data"]["num_workers"],
                          collate_fn=collate, pin_memory=torch.cuda.is_available(),
                          persistent_workers=True, prefetch_factor=4)
    val_dl   = DataLoader(val_ds, batch_size=cfg["data"]["val_batch_size"],
                          shuffle=False, num_workers=cfg["data"]["num_workers"],
                          collate_fn=collate, pin_memory=torch.cuda.is_available(),
                          persistent_workers=True, prefetch_factor=4)
    print("train_ds.cached:", getattr(train_ds, "cached", False))
    print("val_ds.cached:", getattr(val_ds, "cached", False))


    # --- Model ---
    mcfg = cfg["model"]
    model = RecipeTransformer(
        vocab_size=tok.vocab_size(),
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        max_len=mcfg["max_len"],
        p=mcfg["dropout"],                # keep using 'dropout' from YAML
        use_rope=mcfg.get("use_rope", True),  # add this line
        tie_weights=mcfg.get("tie_weights", True),
    ).to(device)

        # Confirm first parameter location
    model_dev = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model device: {model_dev} | params: {n_params/1e6:.2f}M")

    # --- Optimizer / AMP / Loss / Accum ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = GradScaler(enabled=(cfg["optim"]["mixed_precision"] and device == "cuda"))
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
    
    # --- accumulation + schedule ---
    accum = int(cfg["optim"].get("grad_accum_steps", 1))
    assert accum >= 1

    steps_per_epoch = len(train_dl)
    opt_steps_per_epoch = (steps_per_epoch + accum - 1) // accum  
    total_opt_steps = opt_steps_per_epoch * int(cfg["optim"]["epochs"])

    # --- Auto warmup calculation ---
    if isinstance(cfg["optim"].get("warmup_steps"), str) and cfg["optim"]["warmup_steps"].lower() == "auto":
        warmup_steps = int(max(500, min(2000, 0.05 * total_opt_steps)))
        print(f"[auto] warmup_steps = {warmup_steps} (≈5% of total {total_opt_steps} steps)")
        cfg["optim"]["warmup_steps"] = warmup_steps
    else:
        warmup_steps = int(cfg["optim"]["warmup_steps"])

    base_lr = float(cfg["optim"]["lr"])

    optimizer_step = 0 
    start_epoch = 0  
    global_step = 0 
    resume_path = cfg.get("resume_path")  # allow via config/override
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
        e, s, opt_step = load_checkpoint(
            Path(resume_path), model=model, optimizer=opt, scaler=scaler, map_location=device
        )
        start_epoch = e
        global_step = s
        optimizer_step = opt_step          
        print(f"→ resumed at epoch={start_epoch}, global_step={global_step}, optimizer_step={optimizer_step}")
    

    early = EarlyStopping(patience=3, delta=0.0, ckpt_dir=ckpt_dir)

    drive_dir = os.environ.get("DRIVE_CKPT_DIR", "").strip()
    backup_every = int(os.environ.get("BACKUP_EVERY", "1"))
    if drive_dir:
        drive_dir = Path(drive_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
        print(f"[backup] Drive checkpoint dir: {drive_dir.resolve()}")
    else:
        print("[backup] DRIVE_CKPT_DIR not set; auto-backup disabled.")

     # ----- Training -----
    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        epoch_start = time.time()
        model.train()
        running = 0.0
        last_lr = None

        opt.zero_grad(set_to_none=True) 
        pbar = tqdm(train_dl, total=len(train_dl),
                    desc=f"train e{epoch+1}", dynamic_ncols=True, 
                    leave=False, mininterval=0.5
        )
        # ----- Training -----
        for batch_idx, (x, y) in enumerate(pbar):
            global_step += 1

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=cfg["optim"]["mixed_precision"]):
                logits = model(x)
                loss = loss_fn(logits.transpose(1, 2), y)
                loss = loss / accum  # normalize for accumulation

            scaler.scale(loss).backward()
            running += loss.item() * accum  # track true (un-divided) batch loss

            # step every `accum` micro-batches OR at end of dataloader
            do_step = (global_step % accum == 0) or (batch_idx == steps_per_epoch - 1)
            if do_step:
                # LR schedule per OPTIMIZER step 
                step_index = optimizer_step + 1  # 1-based for warmup math clarity
                lr = cosine_schedule(step_index, warmup_steps, total_opt_steps, base_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                last_lr = lr

                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                optimizer_step += 1


            # show live stats in the bar
            if (batch_idx % 20) == 0:
                pbar.set_postfix(
                loss=f"{(running/(batch_idx+1)):.3f}",
                lr=f"{(last_lr or base_lr):.2e}",
                )
            if global_step % 500 == 0: 
                true_mb_loss = loss.item() * accum
                writer.add_scalar("loss/train_step", true_mb_loss, global_step)
                writer.add_scalar("lr/step", (last_lr or base_lr), global_step)
                
                writer.flush()

        avg_train = running / max(1, len(train_dl))

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vpbar = tqdm(val_dl, total=len(val_dl), desc=f"valid e{epoch+1}",
                        dynamic_ncols=True, leave=False)
            for x, y in vpbar:
                x, y = x.to(device), y.to(device)
                with autocast(device_type="cuda", enabled=cfg["optim"]["mixed_precision"]):
                    logits = model(x)
                    vloss  = loss_fn(logits.transpose(1, 2), y)
                val_loss += vloss.item()
                vpbar.set_postfix(vloss=f"{vloss.item():.3f}")


        avg_val = val_loss / max(1, len(val_dl))
        val_ppl = math.exp(min(20, avg_val))
        epoch_time = time.time() - epoch_start

        # Print
        print(f"epoch {epoch+1}: train loss={avg_train:.4f}, val loss={avg_val:.4f}, "
            f"ppl={val_ppl:.2f}, lr={(last_lr or base_lr):.6g}, t={epoch_time:.1f}s")
        # --- Throughput measurement (approximate tokens/sec) ---
        tokens_per_sample = x.size(1) - 1              
        seen_samples = len(train_dl) * cfg["data"]["train_batch_size"]
        seen_tokens = seen_samples * tokens_per_sample
        tps = seen_tokens / max(1e-6, epoch_time)
        print(f"~{tps/1e6:.2f}M tokens/sec (approx)")

        # TensorBoard
        writer.add_scalar("loss/train", avg_train, epoch + 1)
        writer.add_scalar("loss/val", avg_val, epoch + 1)
        writer.add_scalar("metrics/perplexity_val", val_ppl, epoch + 1)
        writer.add_scalar("lr", (last_lr or base_lr), epoch + 1)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch + 1)
        writer.add_text("samples/chicken_garlic_onion", quick_generate(model, tok, device), epoch + 1)
        writer.add_scalar("speed/tokens_per_sec", tps, epoch + 1)

        writer.flush()

        # CSV
        csv_writer.writerow([epoch + 1, f"{avg_train:.6f}", f"{avg_val:.6f}",
                            f"{val_ppl:.6f}", f"{(last_lr or base_lr):.8f}",
                            f"{epoch_time:.2f}"])
        csv_f.flush()

        # Save "last"
        save_checkpoint(
            ckpt_dir / "last_model.pt",
            model=model, optimizer=opt, scaler=scaler,
            epoch=epoch+1, step=global_step, optimizer_step=optimizer_step, cfg=cfg
        )

        # Early stopping
        improved, stop = early.step(avg_val, model)
        if improved:
            save_checkpoint(
                ckpt_dir / "best_model.pt",
                model=model, optimizer=opt, scaler=scaler,
                epoch=epoch+1, step=global_step, optimizer_step=optimizer_step, cfg=cfg
            )
        
        # Optional: auto-backup to Drive if env set
        if drive_dir and ((epoch + 1) % backup_every == 0):
            to_copy = ("last_model.pt", "best_model.pt", "training_log.csv")
            for name in to_copy:
                src = ckpt_dir / name
                dst = drive_dir / name
                if src.exists():
                    try:
                        shutil.copy2(src, dst)
                        print(f"[backup] Copied {src} -> {dst}")
                    except Exception as e:
                        print(f"[backup][ERROR] Failed copying {src} -> {dst}: {e}")
                else:
                    print(f"[backup] Skip: {src} not found")



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
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_small.yaml")
    p.add_argument("--override", type=str, default=None, help='comma list like "optim.lr=1e-3,model.d_model=256"')
    p.add_argument("--resume", type=str, default=None, help='Path to checkpoint (e.g. "model/checkpoints/last_model.pt").')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.resume:
        cfg["resume_path"] = args.resume
    cfg = apply_overrides(cfg, args.override)
    train(cfg)

