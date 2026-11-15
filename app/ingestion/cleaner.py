
import re

from app.ingestion.loader import Document


def clean_text(text: str) -> str:
    # Fix hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common header/footer patterns
    text = re.sub(
        r"(SKCET|Sri Krishna College).*?\n",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Strip whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def clean_document(doc: Document) -> Document:
    return Document(
        text=clean_text(doc.text),
        source=doc.source,
        page_count=doc.page_count,
        metadata=doc.metadata,
    )
