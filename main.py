from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from config import get_settings
from models import IngestResponse, QueryRequest, QueryResponse
from services import RAGService

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting RAG service (provider=%s) ...", settings.llm_provider.value)
    app.state.rag_service = RAGService.create(settings)
    logger.info("RAG service ready.")
    yield


app = FastAPI(
    title="Hybrid Search RAG over Internal Documentation",
    version="1.0.0",
    lifespan=lifespan,
)


def get_rag_service(request: Request) -> RAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready.")
    return service


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health(service: RAGService = Depends(get_rag_service)) -> dict[str, object]:
    return {"status": "ok", "indexed": service.is_indexed}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(service: RAGService = Depends(get_rag_service)) -> IngestResponse:
    try:
        return await service.ingest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    if not service.is_indexed:
        raise HTTPException(status_code=409, detail="No documents indexed yet. Call POST /ingest first.")
    try:
        return await service.query(payload.query, top_n=payload.top_n)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
