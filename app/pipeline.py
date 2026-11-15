from app.ingestion.loader import load_documents
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import chunk_documents
from app.retriever import EmbeddingRetriever
from app.bm25 import BM25Retriever
from app.reranker import rerank
from app.generator import generate_answer
from pathlib import Path

def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
    top_n: int = 5
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

class RAGPipeline:
    def __init__(self, doc_dir: Path = Path("data/docs")):
        self.doc_dir = doc_dir
        self.embedding_retriever = EmbeddingRetriever()
        self.bm25_retriever = BM25Retriever()
        self.chunks = []

    def build_index(self) -> dict:
        docs = load_documents(self.doc_dir)
        docs = [clean_document(d) for d in docs]
        self.chunks = chunk_documents(docs, size=150, overlap=30)

        self.embedding_retriever.index(self.chunks)
        self.bm25_retriever.index(self.chunks)

        return {
            "documents": len(docs),
            "chunks": len(self.chunks)
        }

    def search(
        self, query: str,
        k: int = 5,
        candidate_k: int = 20
    ) -> list[tuple[int, float, str]]:
        bm25_results = self.bm25_retriever.search(query, k=candidate_k)
        emb_results = self.embedding_retriever.search(query, k=candidate_k)

        bm25_ranking = [doc_id for doc_id, _, _ in bm25_results]
        emb_ranking = [doc_id for doc_id, _, _ in emb_results]

        fused = reciprocal_rank_fusion(
            [bm25_ranking, emb_ranking], top_n=candidate_k
        )
        reranked = rerank(
            query,
            [(doc_id, score, self.chunks[doc_id].text)
             for doc_id, score in fused],
            top_n=k
        )
        return reranked

    
    def ask(self, query: str, k: int = 3) -> dict:
        results = self.search(query, k=k)

        sources = [
            self.chunks[doc_id].source
            for doc_id, _, _ in results
        ]

        answer = generate_answer(query, results, sources)
        return answer


    def hybrid_search(
    self, query: str,
    k: int = 20,
    rerank: bool = True,
    candidate_k: int = 20
    ) -> list[tuple[int, float, str]]:
        bm25_results = self.bm25_retriever.search(query, k=candidate_k)
        emb_results  = self.embedding_retriever.search(query, k=candidate_k)

        bm25_ranking = [doc_id for doc_id, _, _ in bm25_results]
        emb_ranking  = [doc_id for doc_id, _, _ in emb_results]

        fused = reciprocal_rank_fusion(
            [bm25_ranking, emb_ranking], top_n=candidate_k
        )
        candidates = [
            (doc_id, score, self.chunks[doc_id].text)
            for doc_id, score in fused
        ]

        if rerank:
            from app.reranker import rerank as do_rerank
            return do_rerank(query, candidates, top_n=k)
        return candidates[:k]