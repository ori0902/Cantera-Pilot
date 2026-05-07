"""Orchestrator integration tests — full pipeline from NL query to analysis report.

Doing two queries (one equilibrium, one flame_speed) because running all 10 eval
cases as unit tests would be slow. The eval runner covers the full eval.
"""
import pytest
from pathlib import Path

from orchestrator import run_query
from config import CHROMA_DIR


pytestmark = pytest.mark.skipif(
    not CHROMA_DIR.exists(),
    reason="RAG index required — run `python -m rag.build_index`",
)


def test_end_to_end_equilibrium():
    rec = run_query(
        "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K",
        verbose=False,
    )
    assert rec.execution_success, f"Pipeline failed: {rec.error} / {rec.execution_stderr[:200]}"
    assert rec.spec is not None
    assert rec.spec.sim_type == "equilibrium"
    assert rec.spec.T == pytest.approx(300.0)
    assert rec.execution_result is not None
    assert 2180 < rec.execution_result["T_ad_K"] < 2280
    assert rec.analysis is not None
    assert len(rec.analysis.summary) > 0


def test_end_to_end_flame_speed():
    rec = run_query(
        "laminar flame speed of stoichiometric CH4/air at 1 atm, 300 K",
        verbose=False,
    )
    assert rec.execution_success, f"Pipeline failed: {rec.error}"
    assert rec.spec.sim_type == "flame_speed"
    assert 0.32 < rec.execution_result["flame_speed_m_s"] < 0.45


def test_logs_directory_has_record():
    """Each run should persist record.json for audit."""
    rec = run_query(
        "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K",
        verbose=False,
    )
    # Pull the most recent logs subdir — the one this run created
    from config import LOG_DIR
    dirs = sorted(LOG_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
    latest = dirs[-1]
    assert (latest / "record.json").exists()
    assert (latest / "sim_attempt1.py").exists()
