from dataclasses import dataclass
from app.ingestion.loader import Document

@dataclass
class Chunk:
    text: str
    chunk_id: int
    source: str
    start_word: int
    end_word: int
    metadata: dict

def chunk_document(
    doc: Document,
    size: int = 150,
    overlap: int = 30
) -> list[Chunk]:
    words = doc.text.split()
    chunks = []
    chunk_id = 0
    i = 0

    while i < len(words):
        window = words[i:i + size]
        text = " ".join(window)

        if len(text.strip()) < 20:  # skip near-empty chunks
            i += size - overlap
            continue

        chunks.append(Chunk(
            text=text,
            chunk_id=chunk_id,
            source=doc.source,
            start_word=i,
            end_word=i + len(window),
            metadata={**doc.metadata, "source": doc.source}
        ))

        chunk_id += 1
        i += size - overlap

    return chunks

def chunk_documents(
    docs: list[Document],
    size: int = 150,
    overlap: int = 30
) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, size, overlap)
        # reassign chunk_ids globally
        for chunk in chunks:
            chunk.chunk_id = len(all_chunks)
            all_chunks.append(chunk)
    return all_chunks