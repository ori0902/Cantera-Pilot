"""RAG retriever.

Given a SimulationSpec, build a retrieval query tuned to what CodeGen will need,
pull top-K from Chroma, return formatted text ready for prompt injection.

Returns empty string gracefully if the index doesn't exist — lets the
orchestrator work during early development before build_index has been run.
"""
from __future__ import annotations
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVAL_K, RERANK_K
from schemas import SimulationSpec


COLLECTION_NAME = "cantera_corpus"


@lru_cache(maxsize=1)
def _get_collection():
    if not CHROMA_DIR.exists():
        return None
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )
    except Exception as e:
        print(f"[retriever] failed to open index: {e}")
        return None


def _build_query(spec: SimulationSpec) -> str:
    """Query terms tuned to each sim_type's API surface."""
    extras = {
        "equilibrium":
            "adiabatic flame temperature equilibrate HP set_equivalence_ratio Solution",
        "ignition_delay":
            "ignition delay IdealGasConstPressureReactor ReactorNet advance time integration",
        "flame_speed":
            "laminar flame speed FreeFlame one dimensional premixed solve",
    }
    return f"cantera {spec.fuel} {extras.get(spec.sim_type, spec.sim_type)}"


def retrieve(spec: SimulationSpec, k: int = RERANK_K) -> str:
    """Return top-k relevant chunks formatted for prompt injection.
    Empty string if no index or error."""
    col = _get_collection()
    if col is None:
        return ""

    query = _build_query(spec)
    try:
        results = col.query(
            query_texts=[query],
            n_results=RETRIEVAL_K,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[retriever] query failed: {e}")
        return ""

    docs = results["documents"][0][:k]
    metas = results["metadatas"][0][:k]
    distances = results.get("distances", [[]])[0][:k]

    pieces = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        header = f"[{meta.get('source', '?')} / {meta.get('title', '?')}]"
        if i < len(distances):
            header += f"  (score={1 - distances[i]:.2f})"
        pieces.append(f"{header}\n{doc.strip()}")
    return "\n\n---\n\n".join(pieces)
