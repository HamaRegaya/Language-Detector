"""
FastAPI application for Language Detection.

Serves both the NB pipeline and CharCNN model via REST API,
with a premium frontend at the root endpoint.

Run:
    uvicorn app.main:app --reload --port 8000
"""
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model import ModelManager, LABELS

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
NB_MODEL_PATH = str(BASE_DIR / "trained_pipeline-0.1.0.pkl")
CNN_MODEL_PATH = str(BASE_DIR / "char_cnn_model.pt")
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── Global model manager ──
manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    global manager
    print("Loading models...")
    manager = ModelManager(nb_path=NB_MODEL_PATH, cnn_path=CNN_MODEL_PATH)
    models_loaded = []
    if manager.nb_model is not None:
        models_loaded.append("NB")
    if manager.cnn_model is not None:
        models_loaded.append(f"CharCNN ({manager.device})")
    print(f"✓ Models loaded: {', '.join(models_loaded)}")
    yield


app = FastAPI(
    title="Language Detector API",
    description="Detect the language of text using MultinomialNB or Character-level CNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response schemas ──
class PredictRequest(BaseModel):
    text: str
    model: str = "cnn"  # "nb" or "cnn"


class BatchPredictRequest(BaseModel):
    texts: list[str]
    model: str = "cnn"


class PredictResponse(BaseModel):
    language: str
    confidence: float
    probabilities: dict[str, float]
    model_used: str
    latency_ms: float


# ── API Routes ──
@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict the language of a text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.perf_counter()
    result = manager.predict(req.text, model=req.model)
    latency = (time.perf_counter() - t0) * 1000

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return PredictResponse(
        language=result["language"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_used=req.model,
        latency_ms=round(latency, 2),
    )


@app.post("/api/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    """Predict languages for multiple texts."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    results = []
    for text in req.texts:
        t0 = time.perf_counter()
        result = manager.predict(text, model=req.model)
        latency = (time.perf_counter() - t0) * 1000
        result["model_used"] = req.model
        result["latency_ms"] = round(latency, 2)
        result["text"] = text[:80]
        results.append(result)
    return {"predictions": results}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "nb": manager.nb_model is not None if manager else False,
            "cnn": manager.cnn_model is not None if manager else False,
        },
        "device": manager.device if manager else "unknown",
        "languages": LABELS,
    }


@app.get("/api/models")
async def models_info():
    """List available models and metadata."""
    info = {"models": []}
    if manager and manager.nb_model is not None:
        info["models"].append({
            "id": "nb",
            "name": "Multinomial Naive Bayes",
            "type": "sklearn Pipeline (CountVectorizer + MultinomialNB)",
            "description": "Fast, interpretable baseline. CPU-only.",
        })
    if manager and manager.cnn_model is not None:
        info["models"].append({
            "id": "cnn",
            "name": "Character-level CNN",
            "type": "PyTorch (3-layer Conv1D + GlobalMaxPool)",
            "description": "Deep learning model trained on CUDA. Captures character n-gram patterns.",
            "device": manager.device,
        })
    return info


# ── Static files (frontend) ──
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(str(STATIC_DIR / "index.html"))
