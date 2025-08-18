preprocess:
	python src/preprocess.py

tokenize:
	python src/tokenization/build_tokenizer.py

train:
	python src/training/train.py

generate:
	python src/training/generate.py
