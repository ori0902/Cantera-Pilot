"""Chunk corpus.jsonl and index into ChromaDB with bge-small embeddings.

- Example scripts: line-aware sliding window (code shouldn't cut mid-statement)
- API docstrings: one chunk per member, since they're usually already the right size

Run after build_corpus.py:  python -m rag.build_index
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_LINES


COLLECTION_NAME = "cantera_corpus"


def _approx_tokens(text: str) -> int:
    """bge tokenizer is BPE; rough rule ~4 chars per token for code/prose mix."""
    return len(text) // 4


def chunk_text(text: str, max_tokens: int = CHUNK_SIZE_TOKENS,
               overlap_lines: int = CHUNK_OVERLAP_LINES) -> list[str]:
    """Line-aware sliding window so code doesn't get cut mid-statement."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        current: list[str] = []
        current_tokens = 0
        while j < len(lines) and current_tokens < max_tokens:
            current.append(lines[j])
            current_tokens += _approx_tokens(lines[j]) + 1
            j += 1
        chunks.append("\n".join(current))
        if j >= len(lines):
            break
        i = max(j - overlap_lines, i + 1)
    return chunks


def build_chunks_from_corpus(corpus_path: Path) -> list[dict]:
    chunks: list[dict] = []
    with corpus_path.open() as f:
        for line in f:
            doc = json.loads(line)

            # API docstrings are small — one chunk
            if doc["source"] == "api":
                chunks.append({
                    "id": f"{doc['path']}#0",
                    "text": doc["text"],
                    "metadata": {
                        "source": doc["source"], "path": doc["path"],
                        "category": doc["category"], "title": doc["title"],
                        "chunk_idx": 0,
                    },
                })
                continue

            for idx, piece in enumerate(chunk_text(doc["text"])):
                chunks.append({
                    "id": f"{doc['path']}#{idx}",
                    "text": piece,
                    "metadata": {
                        "source": doc["source"], "path": doc["path"],
                        "category": doc["category"], "title": doc["title"],
                        "chunk_idx": idx,
                    },
                })
    return chunks


def main() -> None:
    corpus_path = DOCS_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        raise SystemExit(
            f"No corpus at {corpus_path}. Run `python -m rag.build_corpus` first."
        )

    # Fresh index every time — avoids stale chunks if the corpus changes
    if CHROMA_DIR.exists():
        print(f"[index] removing existing index at {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[index] chunking corpus...")
    chunks = build_chunks_from_corpus(corpus_path)
    print(f"[index] {len(chunks)} chunks")

    print(f"[index] loading embedding model '{EMBEDDING_MODEL}' (CPU)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"[index]   embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"\n[index] done. Index at {CHROMA_DIR}")
    print(f"[index] collection '{COLLECTION_NAME}' has {col.count()} chunks")


if __name__ == "__main__":
    main()
