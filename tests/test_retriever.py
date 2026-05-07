"""Retriever smoke test.

Requires the RAG index to exist. Build it with:
    python -m rag.build_corpus
    python -m rag.build_index

Then run:
    python -m pytest tests/test_retriever.py -v

Skipped cleanly if index isn't built yet.
"""
import pytest
from pathlib import Path

from config import CHROMA_DIR
from rag import retriever
from schemas import SimulationSpec


pytestmark = pytest.mark.skipif(
    not CHROMA_DIR.exists(),
    reason="RAG index not built yet — run `python -m rag.build_index` first",
)


def _spec(sim_type, fuel="CH4"):
    return SimulationSpec(
        sim_type=sim_type,
        fuel=fuel,
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T=300.0,
        P=101325.0,
    )


def test_retrieval_returns_nonempty_for_equilibrium():
    docs = retriever.retrieve(_spec("equilibrium"))
    assert len(docs) > 100, "Expected substantial retrieval for equilibrium query"


def test_retrieval_mentions_relevant_api_for_equilibrium():
    """We should see set_equivalence_ratio or equilibrate in the retrieved docs."""
    docs = retriever.retrieve(_spec("equilibrium")).lower()
    relevant = any(term in docs for term in [
        "set_equivalence_ratio", "equilibrate", "adiabatic"
    ])
    assert relevant, f"No relevant API terms in retrieval:\n{docs[:500]}"


def test_retrieval_mentions_reactor_for_ignition_delay():
    docs = retriever.retrieve(_spec("ignition_delay", fuel="H2")).lower()
    relevant = any(term in docs for term in [
        "reactor", "reactornet", "advance", "ignition"
    ])
    assert relevant, f"No reactor-related terms:\n{docs[:500]}"


def test_retrieval_mentions_freeflame_for_flame_speed():
    docs = retriever.retrieve(_spec("flame_speed")).lower()
    relevant = any(term in docs for term in [
        "freeflame", "flame speed", "one-dimensional", "premixed", "laminar"
    ])
    assert relevant, f"No flame-speed-related terms:\n{docs[:500]}"


def test_retrieval_results_differ_by_sim_type():
    """Sanity: different sim types should retrieve different content."""
    eq = retriever.retrieve(_spec("equilibrium"))
    fs = retriever.retrieve(_spec("flame_speed"))
    assert eq != fs, "Retrieval is returning identical content regardless of sim_type"
