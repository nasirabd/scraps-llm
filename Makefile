preprocess:
	python -m src.preprocess --raw data/raw/dev_samples.jsonl --format jsonl 

tokenize:
	python src/tokenization/build_tokenizer.py

train:
	python src/training/train.py

generate:
	python src/training/generate.py
