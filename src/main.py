"""
Few-Shot Product Recognition API
==================================
Register product classes with just 5-10 example images each, then classify
new images against those prototypes -- no training loop, no GPU required,
low-latency inference.

POST /classes/{class_name}  (multipart: 5-10 image files)
    -> builds/overwrites the prototype for that class

POST /predict  (multipart: 1 image file)
    -> returns predicted class + confidence + inference latency

GET  /classes
    -> lists registered classes and how many reference images each has

GET  /health
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.embedder import get_embedder
from src.prototype_store import PrototypeStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fewshot_api")

app = FastAPI(
    title="Few-Shot Product Recognition",
    description="Classify product images using 5-10 reference images per class (Prototypical Networks style, no training loop).",
    version="1.0.0",
)

store = PrototypeStore()
STORE_PATH = os.environ.get("PROTOTYPE_STORE_PATH", "prototypes.json")

MIN_REFERENCE_IMAGES = 3
MAX_REFERENCE_IMAGES = 10

# Module-level File(...) defaults to satisfy FastAPI while avoiding function calls
# in parameter defaults (this prevents ruff B008). These are singletons reused
# as default values in endpoint signatures.
REFERENCE_FILES_DEFAULT = File(...)  # for list[UploadFile] in add_class
SINGLE_FILE_DEFAULT = File(...)      # for single UploadFile in predict


@app.on_event("startup")
def load_existing_store():
    if os.path.isfile(STORE_PATH):
        store.load(STORE_PATH)
        logger.info("Loaded existing prototype store with %d classes", len(store.prototypes))


@app.get("/health")
def health():
    return {"status": "ok", "classes_registered": len(store.prototypes)}


@app.get("/classes")
def list_classes():
    return store.list_classes()


@app.post("/classes/{class_name}")
async def add_class(class_name: str, files: list[UploadFile] = REFERENCE_FILES_DEFAULT):
    if not (MIN_REFERENCE_IMAGES <= len(files) <= MAX_REFERENCE_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"Provide between {MIN_REFERENCE_IMAGES} and {MAX_REFERENCE_IMAGES} reference images (got {len(files)}). "
                   f"Few-shot classification is designed around a small, curated set -- more images stop helping quickly "
                   f"and just slow down setup.",
        )

    embedder = get_embedder()
    embeddings = []
    for f in files:
        content = await f.read()
        try:
            embeddings.append(embedder.embed(content))
        except (ValueError, TypeError, OSError) as e:
            # Catch specific, expected error types rather than a blind `except Exception`
            raise HTTPException(status_code=400, detail=f"Could not process image '{f.filename}': {e}")

    store.add_class(class_name, embeddings)
    store.save(STORE_PATH)
    logger.info("Registered class '%s' with %d reference images", class_name, len(embeddings))

    return {
        "class_name": class_name,
        "reference_images_used": len(embeddings),
        "total_classes_registered": len(store.prototypes),
    }


@app.delete("/classes/{class_name}")
def remove_class(class_name: str):
    existed = store.remove_class(class_name)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found.")
    store.save(STORE_PATH)
    return {"removed": class_name}


@app.post("/predict")
async def predict(file: UploadFile = SINGLE_FILE_DEFAULT):
    if not store.prototypes:
        raise HTTPException(status_code=400, detail="No classes registered yet. POST to /classes/{class_name} first.")

    content = await file.read()
    embedder = get_embedder()
    try:
        embedding, latency_ms = embedder.embed_with_timing(content)
    except (ValueError, TypeError, OSError) as e:
        # Catch specific, expected error types rather than a blind `except Exception`
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    result = store.predict(embedding)
    result["embedding_latency_ms"] = round(latency_ms, 2)
    logger.info(
        "Prediction: %s (confidence=%.3f, latency=%.1fms)",
        result.get("predicted_class"), result.get("confidence", 0), latency_ms,
    )
    return JSONResponse(result)
