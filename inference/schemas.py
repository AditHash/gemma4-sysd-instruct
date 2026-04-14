"""Request and response schemas for the inference API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2048, description="Technical question to answer")


class AskResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
