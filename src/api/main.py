from fastapi import FastAPI
from pydantic import BaseModel
import os, torch
from src.infer.generate import load_from_config, generate, build_prompt

APP = FastAPI(title="Scraps-LLM API", version="0.3")

CFG = os.getenv("SCRAPS_CONFIG", "configs/train_small.yaml")
CKPT_DIR = os.getenv("SCRAPS_CKPT_DIR", "model/checkpoints")
VOCAB = os.getenv("SCRAPS_VOCAB", "tokenizer/bpe.json")

_cache = {"model": None, "tok": None, "device": "cpu"}

def ensure_loaded():
    if _cache["model"] is None:
        model, tok, device = load_from_config(CFG, ckpt_dir=CKPT_DIR, vocab_path=VOCAB, device=None)
        _cache.update(model=model, tok=tok, device=device)
    return _cache["model"], _cache["tok"], _cache["device"]

class GenRequest(BaseModel):
    ingredients: str
    max_new_tokens: int = 140
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    scaffold: str = "basic"

@APP.get("/health")
def health():
    try:
        ensure_loaded()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@APP.post("/generate")
def post_generate(req: GenRequest):
    model, tok, device = ensure_loaded()
    prompt = build_prompt(req.ingredients, req.scaffold)
    full, cont = generate(
        model, tok, prompt, device=device,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        stop_at_eos=True
    )
    cont = cont.replace("Ġ"," ").replace("  "," ").strip()
    return {"ingredients": req.ingredients, "recipe": cont}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SCRAPS_PORT", "5000"))
    uvicorn.run("main:APP", host="0.0.0.0", port=port, reload=False)