import argparse
from pathlib import Path
import torch
import yaml

from src.model.transformer import RecipeTransformer
from src.tokenization.tokenizer import BPETok

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_ckpt_path(ckpt_arg: str) -> Path:
    """
    Accepts either:
      - a path to a file (e.g., best_model.pt), or
      - a directory containing best_model.pt / last_model.pt
    """
    p = Path(ckpt_arg)
    if p.is_file():
        return p
    if p.is_dir():
        best = p / "best_model.pt"
        last = p / "last_model.pt"
        if best.exists():
            return best
        if last.exists():
            return last
    raise FileNotFoundError(
        f"Could not resolve checkpoint from '{ckpt_arg}'. "
        f"Pass a file (…/*.pt) or a dir containing best_model.pt / last_model.pt."
    )

def _extract_state_dict(obj):
    """
    Supports common save formats:
      - torch.save(model.state_dict())
      - torch.save({'model': model.state_dict(), ...})
      - torch.save({'state_dict': model.state_dict(), ...})
    """
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    # assume raw state_dict
    return obj

def build_model_and_load_ckpt(cfg_path: str, ckpt_arg: str, map_location="cpu") -> tuple[RecipeTransformer, BPETok]:
    cfg = load_cfg(cfg_path)
    tok = BPETok(cfg.get("vocab_path", "tokenizer/bpe.json"))

    mcfg = cfg["model"]
    model = RecipeTransformer(
        vocab_size=tok.vocab_size(),
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        max_len=mcfg["max_len"],
        p=mcfg.get("dropout", 0.0),
        use_rope=mcfg.get("use_rope", True),
        tie_weights=mcfg.get("tie_weights", True),
    )

    ckpt_path = _resolve_ckpt_path(ckpt_arg)
    payload = torch.load(ckpt_path, map_location=map_location)
    state = _extract_state_dict(payload)

    # Use strict=True if your checkpoint exactly matches; flip to False if EMA keys, etc.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tok

def main():
    ap = argparse.ArgumentParser(description="Export Scraps-LLM to ONNX")
    ap.add_argument("--config", default="configs/train_small.yaml", help="YAML used to build the model")
    ap.add_argument("--ckpt", default="model/checkpoints", help="Path to .pt OR a dir with best_model.pt/last_model.pt")
    ap.add_argument("--out", default="export/scraps.onnx", help="Output ONNX path")
    ap.add_argument("--seq_len", type=int, default=128, help="Dummy input sequence length for export graph")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--check", action="store_true", help="Run a quick ONNXRuntime shape check after export")
    args = ap.parse_args()

    model, tok = build_model_and_load_ckpt(args.config, args.ckpt, map_location="cpu")
    model.to("cpu")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Dummy input: (B=1, T=seq_len) int64 token IDs
    dummy = torch.randint(low=0, high=tok.vocab_size(), size=(1, args.seq_len), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "logits":    {0: "batch", 1: "seq"}},
        do_constant_folding=True,
    )
    print(f"✅ Saved ONNX to {args.out}")

    if args.check:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
            x = dummy.numpy()
            y = sess.run(["logits"], {"input_ids": x})[0]
            print(f"ORT ok. input={x.shape}  logits={y.shape}")
        except Exception as e:
            print("⚠️ ONNXRuntime check failed:", repr(e))

if __name__ == "__main__":
    main()
