"""FastAPI inference endpoint for the fine-tuned Gemma 4 backend/AI expert."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from inference.model_loader import generate, is_loaded, load_model
from inference.schemas import AskRequest, AskResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    logger.info("Starting up — loading model...")
    load_model()
    logger.info("Model ready. API accepting requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Gemma 4 Backend/AI Expert",
    description="Fine-tuned Gemma 4 E4B for backend engineering and AI/ML questions.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a backend/AI engineering question.

    - Max input: 512 tokens (enforced by model_loader)
    - Max output: 1024 tokens
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.perf_counter()
    try:
        answer = generate(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    latency_ms = (time.perf_counter() - start) * 1000

    return AskResponse(
        answer=answer,
        model="gemma4-backend-ai-expert",
        latency_ms=round(latency_ms, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", model_loaded=is_loaded())
