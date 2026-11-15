import pytest
import json
from pathlib import Path
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import Document

def make_doc(text: str) -> Document:
    return Document(
        text=text, source="test.txt",
        page_count=1, metadata={}
    )

def test_clean_removes_page_numbers():
    text = "Some content\n\n42\n\nMore content"
    assert "42" not in clean_text(text)

def test_clean_fixes_hyphenated_breaks():
    text = "col-\nlege"
    assert "college" in clean_text(text)

def test_chunk_overlap():
    words = " ".join([f"word{i}" for i in range(100)])
    doc = make_doc(words)
    chunks = chunk_document(doc, size=20, overlap=5)
    # last word of chunk N should appear in chunk N+1
    last_words = set(chunks[0].text.split()[-5:])
    first_words = set(chunks[1].text.split()[:5])
    assert last_words & first_words

def test_chunk_ids_are_unique():
    doc = make_doc(" ".join(["word"] * 500))
    chunks = chunk_document(doc, size=50, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

def test_chunk_minimum_length():
    doc = make_doc("short")
    chunks = chunk_document(doc, size=150, overlap=30)
    for chunk in chunks:
        assert len(chunk.text.strip()) >= 20


def test_all_questions_labeled():
    with open("eval/tuning.json", encoding="utf-8") as f:
        questions = json.load(f)

    unlabeled = [
        q["id"]
        for q in questions
        if q.get("type") != "negative"
        and not q.get("relevant_chunks")
    ]

    assert not unlabeled, (
        f"Answerable questions without relevant chunks: {unlabeled}"
    )
