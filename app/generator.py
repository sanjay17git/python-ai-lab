
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check the .env file in your project root."
    )

client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-120b"



def generate_answer(
    query: str,
    reranked_results: list[tuple[int, float, str]],
    chunk_sources: list[str]
) -> dict:

    if not reranked_results:
        return {
            "answer": (
                "I cannot answer this question because "
                "no relevant context was retrieved."
            ),
            "citations": []
        }

    context_parts = []
    citations = []

    for (chunk_id, score, text), source_name in zip(
        reranked_results, chunk_sources
    ):
        source_label = f"{source_name}, chunk {chunk_id}"

        context_parts.append(
            f"[{source_label}]\n{text}"
        )

        citations.append({
            "source": source_name,
            "chunk_id": chunk_id,
            "text": text[:200]
        })

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Rules:
1. Answer using ONLY the supplied context.
2. Do not use outside knowledge or invent facts.
3. If the context does not contain enough information, say:
   "I cannot answer this from the provided documents."
4. Cite every factual answer using the exact source labels provided,
   such as [college_rules.txt, chunk 1].
5. Be concise and clear.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer the question using the rules above."
                )
            }
        ],
        temperature=0.1,
        max_completion_tokens=300
    )

    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "citations": citations
    }
