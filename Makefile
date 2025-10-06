# Makefile

# arguments for preprocessing
RAW=data/raw/dev_samples.jsonl
FORMAT=jsonl
OUTDIR=data/processed
SEED=42
TRAIN=0.9
VAL=0.05
TEST=0.05
ING_COL=ingredients
REC_COL=recipe
LOWER_INGREDIENTS=--lower-ingredients

# arguments for generating recipe
CONFIG        ?= configs/train_small.yaml
CKPT_DIR      ?= model/checkpoints
VOCAB         ?=
INGREDIENTS   ?= chicken, garlic, onion
MAX_NEW_TOKENS ?= 200
TEMPERATURE    ?= 0.8
TOP_K          ?= 50
TOP_P          ?= 0.9
SCAFFOLD       ?= title_step1          
NUM_RECIPES    ?= 3
SHOW_FULL     ?=                 
NO_EOS_STOP   ?=                 
JUST_RECIPE   ?= 1 

# arguments for bleu and rouge
EVAL_SPLIT=val
EVAL_LIMIT=200
EVAL_TEMPERATURE=0.7
EVAL_TOP_K=50
EVAL_TOP_P=0.9
EVAL_SAVE_CSV=
EVAL_OUT_DIR=


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
		--lower-ingredients $(LOWER_INGREDIENTS)


tokenize:
	python -m src.tokenization.build_tokenizer

test:
	python -m src.model.test

train:
	python -m src.training.train 

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
	$(MAKE) eval-text EVAL_SPLIT=val EVAL_SAVE_CSV=outputs/val_preds.csv EVAL_OUT_DIR=model\eval\val

eval-test:
	$(MAKE) eval-text EVAL_SPLIT=test EVAL_SAVE_CSV=outputs/test_preds.csv EVAL_OUT_DIR=model\eval\test

eval-all:
	$(MAKE) eval-val
	$(MAKE) eval-test

tb:
	tensorboard --logdir runs --port 6006


# arguments for gradio app

# Defaults (override on the command line: make app APP_PORT=7861 SHARE=1)
APP_PORT ?= 7860
# Set SHARE to 1 / true / yes to add --share
SHARE    ?= 0

# Extra args you might want to pass through to src/app.py (optional)
APP_ARGS ?=


SHARE_FLAG = $(if $(filter 1 true TRUE yes YES,$(SHARE)),--share,)

.PHONY: app app-auto app-share app-port

# Main target: run with an explicit port (default 7860)
app:
	python -m src.app --port $(APP_PORT) $(SHARE_FLAG) $(APP_ARGS)

# Try the app's internal auto-pick/backup list (omit --port so app tries 7860/7861/7862/auto)
app-auto:
	python -m src.app $(SHARE_FLAG) $(APP_ARGS)

# Convenience: force sharing (public gradio.live tunnel) on default port
app-share:
	$(MAKE) app SHARE=1

# Convenience: pick a specific port quickly
# Usage: make app-port PORT=7862
app-port:
	$(MAKE) app APP_PORT=$(PORT)



