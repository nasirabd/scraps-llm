import torch, yaml, bitsandbytes as bnb
from pathlib import Path
from src.model.transformer import RecipeTransformer
from src.tokenization.tokenizer import BPETok

def load_cfg(p): return yaml.safe_load(open(p, "r", encoding="utf-8"))

def convert_linear_8bit(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            qlin = bnb.nn.Linear8bitLt(module.in_features, module.out_features, bias=(module.bias is not None))
            qlin.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                qlin.bias.data.copy_(module.bias.data)
            parent = model
            *path, leaf = name.split(".")
            for p in path: parent = getattr(parent, p)
            setattr(parent, leaf, qlin)
    return model

def main(cfg_path="configs/train_small.yaml", ckpt_dir="model/checkpoints", out="export/scraps_bnb8.pt"):
    cfg = load_cfg(cfg_path)
    tok = BPETok(cfg["vocab_path"])
    mcfg = cfg["model"]
    model = RecipeTransformer(
        vocab_size=tok.vocab_size(),
        d_model=mcfg["d_model"], n_layers=mcfg["n_layers"], n_heads=mcfg["n_heads"],
        max_len=mcfg["max_len"], p=mcfg.get("dropout", 0.0), tie_weights=mcfg.get("tie_weights", True)
    )
    ckpt = Path(ckpt_dir, "best_model.pt")
    if not ckpt.exists(): ckpt = Path(ckpt_dir, "last_model.pt")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    model = convert_linear_8bit(model)
    torch.save({"model": model.state_dict()}, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
