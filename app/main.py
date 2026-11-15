import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = RAGPipeline(Path("data/docs"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading pipeline...")
    pipeline.build_index()
    logger.info(f"Pipeline ready: {len(pipeline.chunks)} chunks")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="RetrievalLab",
    description="Hybrid RAG over SKCET Code of Conduct",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ──────────────────────────────

class AskRequest(BaseModel):
    query: str
    k: int = 3

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class ChunkResult(BaseModel):
    chunk_id: int
    score: float
    text: str
    source: str

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[dict]
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    latency_ms: float

class IndexResponse(BaseModel):
    documents: int
    chunks: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    pipeline_ready: bool

# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        chunks_indexed=len(pipeline.chunks),
        pipeline_ready=len(pipeline.chunks) > 0
    )

@app.post("/index", response_model=IndexResponse)
async def index_documents():
    try:
        start = time.perf_counter()
        result = pipeline.build_index()
        latency = (time.perf_counter() - start) * 1000
        return IndexResponse(
            documents=result["documents"],
            chunks=result["chunks"],
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(req: AskRequest):
    if not pipeline.chunks:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Call /index first."
        )
    try:
        start = time.perf_counter()
        results = pipeline.search(req.query, k=req.k)
        latency = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=req.query,
            results=[
                ChunkResult(
                    chunk_id=doc_id,
                    score=round(score, 4),
                    text=text,
                    source=pipeline.chunks[doc_id].source
                )
                for doc_id, score, text in results
            ],
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not pipeline.chunks:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Call /index first."
        )
    try:
        # retrieval
        t0 = time.perf_counter()
        results = pipeline.search(req.query, k=req.k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # generation
        t1 = time.perf_counter()
        answer = pipeline.ask(req.query, k=req.k)
        generation_ms = (time.perf_counter() - t1) * 1000

        total_ms = retrieval_ms + generation_ms

        return AskResponse(
            query=req.query,
            answer=answer["answer"],
            citations=answer["citations"],
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2)
        )
    except Exception as e:
        logger.error(f"Ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))