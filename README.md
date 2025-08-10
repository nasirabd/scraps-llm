# Scraps-LLM 🍲

*"Because every leftover has a delicious story to tell"*  

Scraps-LLM is your AI kitchen companion. Give it a list of whatever’s hiding in your fridge — from last night’s roast chicken to that lonely half-onion — and it will create a step-by-step recipe just for you. It’s all about turning food scraps into feasts, reducing waste, and adding a dash of creativity to your cooking.

---

## 🛠 Under the Hood
Scraps-LLM is an **autoregressive, decoder-only Transformer** built in **PyTorch** that generates recipes token-by-token from an ingredient list.  
The project demonstrates:
- **Custom Tokenizer Training** with Byte Pair Encoding (BPE)
- **Causal Masking** for next-token prediction
- **Evaluation** via BLEU and perplexity
- **Dockerized Workflows** for reproducible training & inference
- **Cloud Readiness** with optional AWS S3/ECR + Kubernetes deployment
- **Large-Scale Preprocessing** with optional Apache Spark

Whether you’re here for the recipes or the architecture, Scraps-LLM serves up both.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/scraps-llm.git
cd scraps-llm
```
### 2. Build Docker images
Before building, ensure you have **Docker** and **Make** installed.

Using Make (recommended):
```bash
make build-train   # training/preprocessing image
make build-infer   # inference API image

Without Make:
```bash
docker build -f docker/Dockerfile.train -t scraps/train .
docker build -f docker/Dockerfile.infer -t scraps/infer .
```
### 3. Preprocess dataset:
```bash
make preprocess
```
### 4. Train the model (Colab or local GPU):
```bash
python src/training/train.py
```
### 5. Serve locally(Fast-API):
Launch the API server:
```bash
make serve

Test the health endpoint:
```bash
curl http://localhost:8080/health
```

### 6. Generate a Recipe:
Send a POST request:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"ingredients":"chicken, garlic, onion"}'
```

### 7. Run a Gradio Web Demo locally
```bash
pip install gradio
python src/app.py
```

## 🌐 Live Demo
### Run a live Demo on Hugging Face Spaces:
Try Scraps-LLM in your browser — no installation needed:

