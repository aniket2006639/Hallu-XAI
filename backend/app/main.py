import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routes import analyze
from app.pipeline.orchestrator import HalluPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hallu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting pipeline ...")
    app.state.pipeline = HalluPipeline()
    if config.PRELOAD_MODELS:
        logger.info("Preloading models (first run downloads ~250 MB, please wait) ...")
        app.state.pipeline.warmup()
    logger.info("Ready.")
    yield


app = FastAPI(title="Hallucination XAI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://.*|http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health")
def health():
    pipeline = app.state.pipeline
    from app.pipeline.retrieval import is_embedder_loaded

    return {
        "status": "ok",
        "classifier": pipeline.classifier.meta.get("source"),
        "kb_passages": len(pipeline.retriever.passages),
        "nli_loaded": pipeline.verifier.is_loaded,
        "embedding_loaded": is_embedder_loaded(),
    }
