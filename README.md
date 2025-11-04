# Scraps-LLM 🍲

*"Because every leftover has a delicious story to tell"*  

Scraps-LLM is your AI kitchen companion. Give it a list of whatever’s hiding in your fridge — from last night’s roast chicken to that lonely half-onion — and it will create a step-by-step recipe just for you. It’s all about turning food scraps into feasts, reducing waste, and adding a dash of creativity to your cooking.

---

## 🛠 Under the Hood
Scraps-LLM is an **autoregressive, decoder-only Transformer** built in **PyTorch** that generates recipes token-by-token from an ingredient list.  
The project demonstrates:
- 🍳 **Autoregressive Decoder-Only Transformer** (PyTorch)
- 🔡 **Custom Byte-Pair Encoding (BPE) Tokenizer**
- 🧠 **Causal Masking** for next-token prediction
- 🧾 **BLEU & ROUGE Evaluation**
- 🐳 **Dockerized Workflows** for reproducible inference/training
- ☁️ **Cloud Ready:** ONNX export, Colab GPU, and Hugging Face Space demo

Whether you’re here for the recipes or the architecture, Scraps-LLM serves up both.

---

## 🧠 Model Overview

| Category | Details |
|-----------|----------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | ≈ 137M |
| **Hidden Dim (d_model)** | 768 |
| **Layers** | 12 |
| **Heads** | 12 |
| **Max Seq Len** | 256 |
| **Dropout** | 0.1 |
| **Use RoPE** | ✅ |
| **Tie Weights** | ✅ |
| **Framework** | PyTorch → ONNX |

---

## 📚 Dataset

| Field | Details |
|-------|----------|
| **Source** | [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) |
| **Tokens Processed** | ~2M |
| **Preprocessing** | Ingredient normalization, Unicode cleaning, de-duplication, lowercase, and BPE encoding |
| **Vocabulary** | ~8K tokens |
| **Special Tokens** | `<bos>`, `<eos>` |

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| **Optimizer** | AdamW |
| **Learning Rate** | 3e-4 |
| **Weight Decay** | 0.01 |
| **Betas** | (0.9, 0.95) |
| **Grad Clip** | 1.0 |
| **Scheduler** | Cosine |
| **Warmup Steps** | Auto |
| **Mixed Precision** | ✅ (FP16) |
| **Epochs** | 3 |
| **Seed** | 42 |
| **Device** | Colab Pro+ A100 GPU |
| **Training Time** | ~4–5 hrs |

---

## 📈 Evaluation Results (Validation)

| Metric | Score |
|---------|-------|
| **Perplexity** | ~7.1 |
| **BLEU-1** | 37.1 |
| **ROUGE-L F1** | 6.7 |

---


## 🚀 How to Run 

### 1. Clone and install requirements:
```bash
git clone https://github.com/nasirabd/scraps-llm.git
cd scraps-llm
pip install -r requirements.txt
Make sure Makefile is installed.
```

### 2. Preprocess dataset:
```bash
make preprocess
```
### 3. Local Training and Inference (Optional Tensorboard tracking):
Training:
```bash
make train
make tb
# resume from last checkpoint
make resume
# resume from best checkpoint
make resume_best
```
Inference:
```bash
make recipe
# for prompt and recipe
make full recipe
```
Inference with Gradio app:
bash
```
make app
```

### 4. 🧪 Local Evaluation:
Run ROUGE-L F1 and BLEU on val/test:
```bash
make eval-all
```

### 5. Inference with Docker and Fast-API
Before building, ensure you have **Docker** .
```bash
make build-infer   # inference API image
make run-infer
# Quick health & sample curl (requires run-infer running)
make api-health
make api-curl
```
### 6. Cloud Training and Inference:
Train the model Collab pro:
run scrap_collab.ipynb on collab

🌐 Live Demo on Hugging Face Free paces:
Try Scraps-LLM in your browser — no installation needed:

Option A: Use my Space
👉https://huggingface.co/spaces/donribbs/scraps-llm-demo

1. Go to [**Hugging Face → Spaces**](https://huggingface.co/spaces)
2. Click **New Space**
3. Set:
   - **SDK:** `Gradio`
   - **App File:** `app.py`
   - **Visibility:** `Public`
4. Commit your code & model artifacts:
   - `export/scraps.onnx`
   - `tokenizer/bpe.json`
5. Hugging Face will automatically build and host your app 🎉

## 🧮 Example Generation
Input: 

Ingredients: chicken, garlic, onion

Output:

title: delicious chicken
- step 1: wash chicken and place in large pot.
- step 2: cover with water and bring to a boil.
- step 3: lower heat and simmer for 1 hour.
- step 4: let chicken cool and debone.
- step 5: place chicken back in pot and add garlic and onion.
- step 6: let simmer for 30 minutes.
- step 7: serve with rice.

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Framework** | PyTorch 2.x |
| **Interface** | Gradio 4.x |
| **Tokenizer** | Hugging Face Tokenizers |
| **Export** | ONNX Runtime |
| **Deployment** | Docker / Hugging Face Spaces |
| **Training Platform** | Google Colab Pro+ (A100) |


## Citation
If you use Scraps-LLM in your research or projects, please cite:

@software{Abdallah2025ScrapsLLM,
  author = {Nasir Abdallah},
  title  = {Scraps-LLM: A Recipe Generation Transformer},
  year   = {2025},
  url    = {https://huggingface.co/donribbs/scraps-llm-model}
}

## 🧾 License
MIT License © Nasir Abdallah

## 🔗 Links

| Platform | Repository |
|-----------|-------------|
| **🤗 Hugging Face Model** | [donribbs/scraps-llm-model](https://huggingface.co/donribbs/scraps-llm-model) |
| **🌐 Hugging Face Space (Demo)** | [donribbs/scraps-llm-demo](https://huggingface.co/spaces/donribbs/scraps-llm-demo) |
| **💻 GitHub Source** | [nasirabd/scraps-llm](https://github.com/nasirabd/scraps-llm) |

