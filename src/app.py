
import os
import torch
import gradio as gr


from src.infer.generate import load_from_config, generate, build_prompt

# --------- Defaults (safe to tweak or override via env) ----------
DEFAULT_CONFIG   = os.getenv("SCRAPS_CONFIG", "configs/train_small.yaml")
DEFAULT_CKPT_DIR = os.getenv("SCRAPS_CKPT_DIR", "model/checkpoints")
DEFAULT_VOCAB    = os.getenv("SCRAPS_VOCAB", "") or None

# Cache the model/tokenizer on first use so app is snappy
_model_tok_cached = {"model": None, "tok": None, "device": None, "cfg_path": None, "ckpt_dir": None, "vocab": None}

def _ensure_loaded(config_path=DEFAULT_CONFIG, ckpt_dir=DEFAULT_CKPT_DIR, vocab_path=DEFAULT_VOCAB):
    ct = _model_tok_cached
    if (ct["model"] is None 
        or ct["cfg_path"] != config_path 
        or ct["ckpt_dir"] != ckpt_dir 
        or ct["vocab"] != vocab_path):
        model, tok, device = load_from_config(config_path, ckpt_dir=ckpt_dir, vocab_path=vocab_path, device=None)
        ct.update(model=model, tok=tok, device=device, cfg_path=config_path, ckpt_dir=ckpt_dir, vocab=vocab_path)
    return ct["model"], ct["tok"], ct["device"]

def _format_recipe(text: str) -> str:
    """Light formatting for nicer display in Markdown."""
    if not text.strip():
        return "_(empty output)_"
    # promote simple lines to markdown sections if present
    lines = text.strip().splitlines()
    out = []
    for ln in lines:
        if ln.strip().lower().startswith("title"):
            out.append(f"### {ln.strip()}")
        elif ln.strip().lower().startswith("step"):
            out.append(f"- {ln.strip()}")
        else:
            out.append(ln)
    return "\n".join(out)

def generate_handler(
    ingredients: str,
    num_recipes: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    no_eos_stop: bool,
    show_full: bool,
    config_path: str,
    ckpt_dir: str,
    vocab_path: str
):
    ingredients = ingredients.strip()
    if not ingredients:
        return "Please enter some ingredients.", ""

    # Load (cached) model + tokenizer
    model, tok, device = _ensure_loaded(
        config_path or DEFAULT_CONFIG,
        ckpt_dir or DEFAULT_CKPT_DIR,
        vocab_path or DEFAULT_VOCAB,
    )

    prompt = build_prompt(ingredients, scaffold)
    recipes_md = []
    fulls_md = []

    for i in range(max(1, int(num_recipes))):
        full, cont = generate(
            model, tok, prompt,
            device=device,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            stop_at_eos=(not no_eos_stop),
        )
        recipes_md.append(f"**Recipe {i+1}**\n\n{_format_recipe(cont)}")
        if show_full:
            fulls_md.append(f"**Full {i+1}**\n\n```\n{full}\n```")

    recipes_panel = "\n\n---\n\n".join(recipes_md)
    full_panel = "\n\n".join(fulls_md) if show_full else ""
    return recipes_panel, full_panel


# -------------- UI ----------------
with gr.Blocks(title="Scraps-LLM: Recipe Generator") as demo:
    gr.Markdown("# 🥘 Scraps-LLM\n_Generate recipes from leftover ingredients._")

    with gr.Row():
        with gr.Column(scale=2):
            ingredients = gr.Textbox(
                label="Ingredients (comma-separated)",
                placeholder="e.g., chicken, garlic, onion",
                lines=2,
                value="chicken, garlic, onion",
            )

            num_recipes = gr.Slider(
                1, 5, value=1, step=1,
                label="Number of recipes to generate"
            )

    with gr.Accordion("Advanced generation settings", open=False):
        with gr.Row():
            max_new_tokens = gr.Slider(32, 512, value=200, step=8, label="Max new tokens")
            temperature    = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
        with gr.Row():
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="Top-p (nucleus)")
        with gr.Row():
            no_eos_stop = gr.Checkbox(False, label="Ignore EOS (generate to max tokens)")
            show_full   = gr.Checkbox(False, label="Also show full decoded text (debug)")

    with gr.Accordion("Model paths (optional overrides)", open=False):
        config_path = gr.Textbox(label="Config path", value=DEFAULT_CONFIG)
        ckpt_dir    = gr.Textbox(label="Checkpoint dir", value=DEFAULT_CKPT_DIR)
        vocab_path  = gr.Textbox(label="Tokenizer path (optional override)", value=DEFAULT_VOCAB or "")

    run_btn = gr.Button("🍳 Generate", variant="primary")

    recipes_out = gr.Markdown(label="Generated Recipe(s)")
    full_out    = gr.Markdown(label="Full (prompt + recipe)")

    run_btn.click(
        fn=generate_handler,
        inputs=[ingredients, num_recipes, max_new_tokens, temperature, top_k, top_p, no_eos_stop, show_full, config_path, ckpt_dir, vocab_path],
        outputs=[recipes_out, full_out],
        api_name="generate"
    )

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    ap.add_argument("--share", action="store_true", default=(os.getenv("SCRAPS_SHARE","0") in ("1","true","True")))
    args = ap.parse_args()

    prefer = args.port
    candidates = [prefer, 7861, 7862, 0]  # 0 = auto-pick any free port
    last_err = None

    for p in candidates:
        try:
            # If p == 0, let Gradio choose any free port by passing None
            res = demo.launch(
                server_name="127.0.0.1",
                server_port=(None if p == 0 else p),
                share=args.share,
                prevent_thread_lock=True,  # non-blocking
                inbrowser=False,
            )
            block = getattr(res, "block", None)
            if callable(block):
                block()
            else:
                # Fallback: keep process alive in a simple loop
                import time
                print("[Gradio] Press Ctrl+C to stop")
                while True:
                    time.sleep(3600)
            break
        except OSError as e:
            last_err = e
            print(f"[Gradio] Port {p if p else '(auto)'} failed: {e}")
    else:
        raise SystemExit(f"[Gradio] Failed to launch. Last error: {last_err}")



