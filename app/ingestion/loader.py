from pathlib import Path
from dataclasses import dataclass
import pdfplumber
import logging
logger = logging.getLogger(__name__)

@dataclass
class Document:
    text: str
    source: str        # filename
    page_count: int
    metadata: dict

def load_pdf(path: Path) -> Document:
    pages = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
            else:
                logger.warning(
                    f"{path.name} page {page.page_number} "
                    f"returned no text — likely image-based"
                )

    if not pages:
        raise ValueError(
            f"{path.name} yielded no extractable text. "
            f"Is it a scanned PDF?"
        )

    return Document(
        text="\n\n".join(pages),
        source=path.name,
        page_count=page_count,
        metadata={"path": str(path)}
    )

def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        text=text,
        source=path.name,
        page_count=1,
        metadata={"path": str(path)}
    )

def load_documents(doc_dir: Path) -> list[Document]:
    loaders = {".pdf": load_pdf, ".txt": load_txt}
    docs = []

    for path in sorted(doc_dir.iterdir()):
        if path.suffix.lower() not in loaders:
            logger.info(f"Skipping unsupported file: {path.name}")
            continue
        try:
            doc = loaders[path.suffix.lower()](path)
            logger.info(
                f"Loaded {path.name}: "
                f"{doc.page_count} pages, "
                f"{len(doc.text)} chars"
            )
            docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")

    return docs