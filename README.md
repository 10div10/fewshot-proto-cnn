# Few-Shot Product Recognition (Prototypical Networks style)

A FastAPI service that classifies product images using only **5-10 example
images per class** — no training loop, no GPU, no thousands of labeled
images. New classes can be registered in seconds; predictions run in
single-digit milliseconds on CPU.

## The core idea

Instead of training a CNN classifier from scratch (which needs hundreds or
thousands of images per class to generalize), this uses a **pretrained**
CNN (MobileNetV3-Small, trained on ImageNet) purely as a fixed feature
extractor. It already knows how to represent visual concepts like edges,
textures, and shapes — we just reuse that representation space instead of
learning a new one.

For each class:
1. Embed each of its 5-10 reference images into a 576-dim feature vector
2. Average those vectors into a single **prototype** — one vector representing the class
3. To classify a new image: embed it, compare to every stored prototype via cosine similarity, return the nearest one

This is the inference-time core of **Prototypical Networks** (Snell et al.,
2017) — no backprop, no epochs, no fine-tuning. Adding a class is O(seconds),
not O(hours).

## Why this over training a full CNN

| | Training a CNN classifier | This (prototype-based) |
|---|---|---|
| Images needed per class | Hundreds–thousands | 5-10 |
| Setup time for a new class | Hours (retrain) | Seconds (embed + average) |
| GPU required | Usually yes | No — CPU inference is ~10ms/image |
| Adding a class later | Full retrain | Just add one more prototype |

The tradeoff: this won't outperform a properly fine-tuned CNN with lots of
data on fine-grained distinctions. It's the right tool when you have very
few examples per class and need to move fast — which is exactly the
"5-10 images instead of 1000" constraint this was built for.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /classes/{class_name}` | Multipart upload of 3-10 reference images → builds/overwrites that class's prototype |
| `DELETE /classes/{class_name}` | Remove a class |
| `GET /classes` | List registered classes and their reference image counts |
| `POST /predict` | Multipart upload of 1 image → predicted class, confidence, top-k, and embedding latency |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

First run will download the pretrained MobileNetV3-Small weights (~10MB,
one-time, cached afterward at `~/.cache/torch`).

## Generate sample data (optional, for a quick demo without real photos)

```bash
python scripts/generate_sample_data.py
```

This creates `sample_data/{red_bottle,blue_mug,green_box}/` with 8
synthetic images each — simple colored shapes standing in for product
photos, useful for testing the pipeline end-to-end without needing a real
product photo dataset.

## Run

```bash
uvicorn src.main:app --reload
```

### Register a class (use 5-6 of the 8 generated images, holding a couple back to test on)

```bash
curl -X POST http://localhost:8000/classes/red_bottle \
  -F "files=@sample_data/red_bottle/red_bottle_0.jpg" \
  -F "files=@sample_data/red_bottle/red_bottle_1.jpg" \
  -F "files=@sample_data/red_bottle/red_bottle_2.jpg" \
  -F "files=@sample_data/red_bottle/red_bottle_3.jpg" \
  -F "files=@sample_data/red_bottle/red_bottle_4.jpg" \
  -F "files=@sample_data/red_bottle/red_bottle_5.jpg"
```

Repeat for `blue_mug` and `green_box` using their respective sample images.

### Classify a held-out image

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_data/red_bottle/red_bottle_6.jpg"
```

Expected response shape:
```json
{
  "predicted_class": "red_bottle",
  "confidence": 0.97,
  "top_k": [
    {"class": "red_bottle", "score": 0.97},
    {"class": "green_box", "score": 0.41},
    {"class": "blue_mug", "score": 0.33}
  ],
  "embedding_latency_ms": 11.2
}
```

## Testing

```bash
pytest tests/ -v
```

8 unit tests cover the prototype math directly (mean-vector construction,
nearest-prototype classification, save/load, edge cases like zero reference
images or no classes registered) — these run with pure numpy, no
torch/network dependency, so they're fast and safe in CI.

The full pipeline (real CNN embeddings + FastAPI) is smoke-tested separately
in CI, since it needs to download pretrained weights on first run.

## Docker

```bash
docker build -t fewshot-proto-cnn .
docker run -p 8000:8000 fewshot-proto-cnn
curl http://localhost:8000/health
```

Pretrained weights are downloaded once at **build time** (baked into the
image), not on the first request — avoids a slow or failing cold start in
production.

## CI/CD

`.github/workflows/ci.yml`:
1. Lints with `ruff`
2. Runs the 8 network-free prototype-store unit tests
3. Runs a full pipeline smoke test (real embeddings on generated sample data)
4. Builds the Docker image and smoke-tests `/health` inside the container

## What I'd add for production

- Vector index (FAISS) instead of a linear scan over prototypes — only
  matters once you have hundreds of classes, linear scan is fine below that
- Confidence thresholding: reject predictions below a similarity cutoff
  instead of always returning a "best guess"
- Persistent storage (S3/blob) for the prototype store instead of a local
  JSON file
- Data augmentation on the reference images (rotation/crop/color jitter)
  before averaging, to make prototypes more robust to real-world photo variation
- Swap MobileNetV3-Small for a larger backbone (ResNet50, EfficientNet) if
  latency budget allows — better embeddings, slower inference

## Project structure

```
fewshot-proto-cnn/
├── src/
│   ├── main.py             # FastAPI app + endpoints
│   ├── embedder.py          # Pretrained MobileNetV3 feature extractor
│   └── prototype_store.py    # Prototype building + nearest-prototype classification
├── scripts/
│   └── generate_sample_data.py  # Synthetic test images (no external downloads needed)
├── tests/
│   └── test_prototype_store.py
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```
