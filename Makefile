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

train:
	python src/training/train.py

generate:
	python src/training/generate.py
