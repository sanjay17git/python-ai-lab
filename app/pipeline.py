import os
from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import chunk_documents
from app.bm25 import BM25Retriever
from app.generator import generate_answer


ENABLE_EMBEDDINGS = os.getenv("ENABLE_EMBEDDINGS", "true").lower() == "true"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"


def reciprocal_rank_fusion(rankings, k=60, top_n=5):
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


class RAGPipeline:
    def __init__(self, doc_dir=Path("data/docs")):
        self.doc_dir = doc_dir
        self.chunks = []

        self.enable_embeddings = ENABLE_EMBEDDINGS
        self.enable_reranker = ENABLE_RERANKER

        # Import SentenceTransformer only when embeddings are enabled.
        if self.enable_embeddings:
            from app.retriever import EmbeddingRetriever
            self.embedding_retriever = EmbeddingRetriever()
        else:
            self.embedding_retriever = None

        self.bm25_retriever = BM25Retriever()

    def build_index(self):
        docs = load_documents(self.doc_dir)
        docs = [clean_document(d) for d in docs]

        self.chunks = chunk_documents(docs, size=150, overlap=30)

        # BM25 is always available.
        self.bm25_retriever.index(self.chunks)

        # Dense embeddings are optional.
        if self.embedding_retriever is not None:
            self.embedding_retriever.index(self.chunks)

        return {
            "documents": len(docs),
            "chunks": len(self.chunks)
        }

    def search(self, query, k=5, candidate_k=20):
        bm25_results = self.bm25_retriever.search(
            query,
            k=candidate_k
        )

        rankings = [
            [doc_id for doc_id, _, _ in bm25_results]
        ]

        # Add semantic retrieval only when enabled.
        if self.embedding_retriever is not None:
            emb_results = self.embedding_retriever.search(
                query,
                k=candidate_k
            )
            rankings.append([
                doc_id for doc_id, _, _ in emb_results
            ])

        fused = reciprocal_rank_fusion(
            rankings,
            top_n=candidate_k
        )

        candidates = [
            (doc_id, score, self.chunks[doc_id].text)
            for doc_id, score in fused
        ]

        # Cross-encoder reranking is optional.
        if self.enable_reranker:
            from app.reranker import rerank as do_rerank
            return do_rerank(query, candidates, top_n=k)

        return candidates[:k]

    def ask(self, query, k=3):
        results = self.search(query, k=k)

        sources = [
            self.chunks[doc_id].source
            for doc_id, _, _ in results
        ]

        answer = generate_answer(query, results, sources)
        return answer

    def hybrid_search(self, query, k=20, rerank=True, candidate_k=20):
        bm25_results = self.bm25_retriever.search(
            query,
            k=candidate_k
        )

        rankings = [
            [doc_id for doc_id, _, _ in bm25_results]
        ]

        if self.embedding_retriever is not None:
            emb_results = self.embedding_retriever.search(
                query,
                k=candidate_k
            )
            rankings.append([
                doc_id for doc_id, _, _ in emb_results
            ])

        fused = reciprocal_rank_fusion(
            rankings,
            top_n=candidate_k
        )

        candidates = [
            (doc_id, score, self.chunks[doc_id].text)
            for doc_id, score in fused
        ]

        if rerank and self.enable_reranker:
            from app.reranker import rerank as do_rerank
            return do_rerank(query, candidates, top_n=k)

        return candidates[:k]
