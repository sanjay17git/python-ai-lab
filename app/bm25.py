import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.ingestion.chunker import Chunk

CACHE_DIR = Path("cache")

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.bm25 = None

    def index(self, chunks) -> None:
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        # rank_bm25 doesn't expose k1/b directly — rebuild with custom scorer
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._save()

    def search(
        self, query: str, k: int = 5
    ) -> list[tuple[int, float, str]]:
        if self.bm25 is None:
            raise RuntimeError("Call index() before search()")
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        return [
            (i, float(scores[i]), self.chunks[i].text)
            for i in indices
        ]

    def _save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / "bm25.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "BM25Retriever":
        with open(CACHE_DIR / "bm25.pkl", "rb") as f:
            return pickle.load(f)