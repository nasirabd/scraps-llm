# =========================
# Scraps-LLM Makefile
# =========================
ifeq ($(OS),Windows_NT)
DOCKER := docker.exe
PATH := C:\Program Files\Docker\Docker\resources\bin;$(PATH)
else
DOCKER := docker
endif

# ---------- Preprocess ----------
RAW            ?= data/raw/recipenlg.csv 
FORMAT         ?= csv
OUTDIR         ?= data/processed
SEED           ?= 42
TRAIN          ?= 0.9
VAL            ?= 0.05
TEST           ?= 0.05

# CSV column names (RecipeNLG default)
ING_COL        ?= ingredients
REC_COL        ?= directions
TITLE_COL      ?= title
NER_COL        ?= NER

# --- Behavior flags ---
# Default: lowercase ingredients, strip quantities, merge with NER (union mode)
LOWER_INGREDIENTS ?= 1
KEEP_QUANTITIES   ?= 0             # default: strip quantities (no numbers/units)
NER_MODE          ?= union         # recommended for good recall coverage

# Portable flag expansions
LOWER_FLAG  := $(if $(filter 1 true TRUE yes YES,$(LOWER_INGREDIENTS)),--lower-ingredients,)
KEEPQ_FLAG  := $(if $(filter 1 true TRUE yes YES,$(KEEP_QUANTITIES)),--keep-quantities,--no-keep-quantities)

# CSV-only flags (auto applied when FORMAT=csv)
CSV_FLAGS := $(if $(filter csv,$(FORMAT)), \
	--ing-col $(ING_COL) \
	--rec-col $(REC_COL) \
	--title-col $(TITLE_COL) \
	--ner-col $(NER_COL) \
	--ner-mode $(NER_MODE),)

# --- Tokenizer params ---
TOK_DATA_DIR     ?= data/processed
TOK_SPLITS       ?= train val test
TOK_FIELDS       ?= ingredients recipe title
TOK_OUT_DIR      ?= tokenizer
TOK_VOCAB_SIZE   ?= 32000
TOK_MIN_FREQ     ?= 2
TOK_LOWER        ?= 1
TOK_NFKC         ?= 1
TOK_SAMPLE_LIMIT ?= 0

TOK_LOWER_FLAG := $(if $(filter 1 true TRUE yes YES,$(TOK_LOWER)),--lower,)
TOK_NFKC_FLAG  := $(if $(filter 1 true TRUE yes YES,$(TOK_NFKC)),--nfkc,)

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
PORT ?= 8080
CONTAINER_PORT ?= 8080
CKPT_HOST ?= "C:\Users\seruc\OneDrive\Desktop\Scraps\scraps-llm\model\checkpoints"
DOCKER_IMAGE ?= scraps/infer
SCRAPS_CONFIG   ?= /app/configs/train_small.yaml
SCRAPS_VOCAB    ?= /app/tokenizer/bpe.json
SCRAPS_CKPT_DIR ?= /app/model/checkpoints
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
		$(CSV_FLAGS) \
		$(LOWER_FLAG) \
		$(KEEPQ_FLAG)

tokenize:
	python -m src.tokenization.build_tokenizer \
	  --data_dir $(TOK_DATA_DIR) \
	  --splits $(TOK_SPLITS) \
	  --fields $(TOK_FIELDS) \
	  --out_dir $(TOK_OUT_DIR) \
	  --vocab_size $(TOK_VOCAB_SIZE) \
	  --min_frequency $(TOK_MIN_FREQ) \
	  $(TOK_LOWER_FLAG) \
	  $(TOK_NFKC_FLAG) \
	  --sample_limit $(TOK_SAMPLE_LIMIT)

cache:
	python -m src.data.cache_tokens

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
	$(DOCKER) build -f docker/Dockerfile.infer -t $(DOCKER_IMAGE) .

run-infer:
	$(DOCKER) run --rm -p $(PORT):$(CONTAINER_PORT) \
		-e SCRAPS_CONFIG=$(SCRAPS_CONFIG) \
		-e SCRAPS_VOCAB=$(SCRAPS_VOCAB) \
		-e SCRAPS_CKPT_DIR=$(SCRAPS_CKPT_DIR) \
		-e SCRAPS_QUANTIZE=$(SCRAPS_QUANTIZE) \
		-v "$(CKPT_HOST):$(SCRAPS_CKPT_DIR)" \
		$(DOCKER_IMAGE)

# Quick health & sample curl (requires run-infer running)
api-health:
	curl -s http://localhost:8080/health | python -m json.tool

api-curl:
	curl -s -X POST http://localhost:8080/generate \
	  -H "Content-Type: application/json" \
	  -d '{"ingredients":"chicken, garlic, onion","max_new_tokens":140,"temperature":0.8,"top_k":50,"top_p":0.9}' \
	  | python -m json.tool
