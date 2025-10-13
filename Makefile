# =========================
# Scraps-LLM Makefile
# =========================

# ---------- Preprocess ----------
RAW            ?= data/raw/dev_samples.jsonl
FORMAT         ?= jsonl
OUTDIR         ?= data/processed
SEED           ?= 42
TRAIN          ?= 0.9
VAL            ?= 0.05
TEST           ?= 0.05
ING_COL        ?= ingredients
REC_COL        ?= recipe
# Use 1 to enable, 0 to disable (portable flag handling)
LOWER_INGREDIENTS ?= 1
LOWER_FLAG := $(if $(filter 1 true TRUE yes YES,$(LOWER_INGREDIENTS)),--lower-ingredients,)

# ---------- Training / Config ----------
CONFIG     ?= configs/train_small.yaml
OVERRIDES  ?=
# example: OVERRIDES=optim.lr=5e-4,model.d_model=256,model.n_heads=8

# ---------- Inference (CLI) ----------
CKPT_DIR       ?= model/checkpoints
VOCAB          ?=
INGREDIENTS    ?= chicken, garlic, onion
MAX_NEW_TOKENS ?= 200
TEMPERATURE    ?= 0.8
TOP_K          ?= 50
TOP_P          ?= 0.9
SCAFFOLD       ?= title_step1
NUM_RECIPES    ?= 3
SHOW_FULL      ?=
NO_EOS_STOP    ?=
JUST_RECIPE    ?= 1

# ---------- Evaluation ----------
EVAL_SPLIT       ?= val
EVAL_LIMIT       ?= 200
EVAL_TEMPERATURE ?= 0.7
EVAL_TOP_K       ?= 50
EVAL_TOP_P       ?= 0.9
EVAL_SAVE_CSV    ?=
EVAL_OUT_DIR     ?=

# ---------- Gradio App ----------
APP_PORT ?= 7860
SHARE    ?= 0
APP_ARGS ?=
SHARE_FLAG = $(if $(filter 1 true TRUE yes YES,$(SHARE)),--share,)

# ---------- Docker (API) ----------
DOCKER_IMAGE ?= scraps/infer
SCRAPS_CONFIG   ?= $(CONFIG)
SCRAPS_VOCAB    ?= tokenizer/bpe.json
SCRAPS_CKPT_DIR ?= model/checkpoints
SCRAPS_QUANTIZE ?= 0

# ---------- Sweeps ----------
SWEEP_LRS      ?= 1e-3 5e-4 3e-4
SWEEP_DMODELS  ?= 128 256
SWEEP_HEADS    ?= 4 8

# ---------- Phony ----------
.PHONY: preprocess tokenize test train sweep recipe full-recipe just-recipe \
        eval-text eval-val eval-test eval-all tb \
        app app-auto app-share app-port \
        build-infer run-infer api-health api-curl

# =========================
# Data
# =========================
preprocess:
	python -m src.preprocess \
		--raw $(RAW) \
		--format $(FORMAT) \
		--outdir $(OUTDIR) \
		--seed $(SEED) \
		--train $(TRAIN) \
		--val $(VAL) \
		--test $(TEST) \
		--ing-col $(ING_COL) \
		--rec-col $(REC_COL) \
		$(LOWER_FLAG)

tokenize:
	python -m src.tokenization.build_tokenizer

test:
	python -m src.model.test

# =========================
# Training
# =========================
train:
	python -m src.training.train --config $(CONFIG) $(if $(OVERRIDES),--override "$(OVERRIDES)")

sweep:
	@for lr in $(SWEEP_LRS); do \
	  for dm in $(SWEEP_DMODELS); do \
	    for h in $(SWEEP_HEADS); do \
	      echo "===> lr=$$lr d_model=$$dm n_heads=$$h"; \
	      python -m src.training.train --config $(CONFIG) \
	        --override "optim.lr=$$lr,model.d_model=$$dm,model.n_heads=$$h"; \
	    done; \
	  done; \
	done

# =========================
# Inference (CLI)
# =========================
recipe:
	python -m src.infer.generate \
		--config $(CONFIG) \
		--ckpt_dir $(CKPT_DIR) \
		$(if $(VOCAB),--vocab $(VOCAB)) \
		--ingredients "$(INGREDIENTS)" \
		--max_new_tokens $(MAX_NEW_TOKENS) \
		--temperature $(TEMPERATURE) \
		--top_k $(TOP_K) \
		--top_p $(TOP_P) \
		--scaffold $(SCAFFOLD) \
		--num_recipes $(NUM_RECIPES) \
		$(if $(SHOW_FULL),--show_full) \
		$(if $(NO_EOS_STOP),--no_eos_stop) \
		$(if $(JUST_RECIPE),--just_recipe)

full-recipe:
	$(MAKE) recipe SHOW_FULL=1

just-recipe:
	$(MAKE) recipe

# =========================
# Evaluation
# =========================
eval-text:
	python -m src.eval.text_eval \
	  --config $(CONFIG) \
	  --ckpt_dir $(CKPT_DIR) \
	  --split $(EVAL_SPLIT) \
	  --limit $(EVAL_LIMIT) \
	  --temperature $(EVAL_TEMPERATURE) \
	  --top_k $(EVAL_TOP_K) \
	  --top_p $(EVAL_TOP_P) \
	  $(if $(EVAL_SAVE_CSV),--save_csv $(EVAL_SAVE_CSV),) \
	  $(if $(EVAL_OUT_DIR),--out_dir $(EVAL_OUT_DIR),)

eval-val:
	$(MAKE) eval-text EVAL_SPLIT=val EVAL_SAVE_CSV=outputs/val_preds.csv EVAL_OUT_DIR=model/eval/val

eval-test:
	$(MAKE) eval-text EVAL_SPLIT=test EVAL_SAVE_CSV=outputs/test_preds.csv EVAL_OUT_DIR=model/eval/test

eval-all:
	$(MAKE) eval-val
	$(MAKE) eval-test

tb:
	tensorboard --logdir runs --port 6006

# =========================
# Gradio App (local)
# =========================
app:
	python -m src.app --port $(APP_PORT) $(SHARE_FLAG) $(APP_ARGS)

app-auto:
	python -m src.app $(SHARE_FLAG) $(APP_ARGS)

app-share:
	$(MAKE) app SHARE=1

# Usage: make app-port PORT=7862
app-port:
	$(MAKE) app APP_PORT=$(PORT)

# =========================
# Docker FastAPI (CPU)
# =========================
build-infer:
	docker build -f docker/Dockerfile.infer -t $(DOCKER_IMAGE) .

run-infer:
	docker run --rm -p 8080:8080 \
		-e SCRAPS_CONFIG=$(SCRAPS_CONFIG) \
		-e SCRAPS_VOCAB=$(SCRAPS_VOCAB) \
		-e SCRAPS_CKPT_DIR=$(SCRAPS_CKPT_DIR) \
		-e SCRAPS_QUANTIZE=$(SCRAPS_QUANTIZE) \
		$(DOCKER_IMAGE)

# Quick health & sample curl (requires run-infer running)
api-health:
	curl -s http://localhost:8080/health | python -m json.tool

api-curl:
	curl -s -X POST http://localhost:8080/generate \
	  -H "Content-Type: application/json" \
	  -d '{"ingredients":"chicken, garlic, onion","max_new_tokens":140,"temperature":0.8,"top_k":50,"top_p":0.9}' \
	  | python -m json.tool
