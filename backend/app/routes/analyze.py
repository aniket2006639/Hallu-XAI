from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import config

router = APIRouter()


class AnalyzeRequest(BaseModel):
    question: str = ""
    answer: str
    use_wikipedia: Optional[bool] = None


@router.post("/analyze")
def analyze(payload: AnalyzeRequest, request: Request):
    answer = payload.answer.strip()
    if not answer:
        raise HTTPException(status_code=422, detail="`answer` must not be empty")
    if len(answer) > config.MAX_ANSWER_CHARS:
        raise HTTPException(status_code=413,
                            detail=f"Text too long ({len(answer)} chars, max {config.MAX_ANSWER_CHARS})")
    pipeline = request.app.state.pipeline
    return pipeline.analyze(payload.question, answer, payload.use_wikipedia)
