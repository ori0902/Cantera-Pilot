"""CodeGen integration tests.

Not testing the exact code the LLM emits — that's stochastic. Testing whether
the CodeGen + Executor + Cantera chain produces the right NUMBER for each
sim type.

Each test builds a SimulationSpec by hand (bypassing Planner), retrieves RAG
context, calls CodeGen, executes the result, and checks the numeric output
against the known reference values we verified in session.

If a test fails on attempt 1 but passes on attempt 2 or 3, the retry loop is
earning its keep. If all three attempts fail, 8B can't reliably write code
for this sim type even with RAG — which is informative.
"""
import pytest

from agents import codegen
from sandbox import executor
from rag import retriever
from schemas import SimulationSpec
from config import CHROMA_DIR


pytestmark = pytest.mark.skipif(
    not CHROMA_DIR.exists(),
    reason="RAG index not built — run `python -m rag.build_index` first",
)


def _make_spec(sim_type, fuel="CH4", T=300.0, P=101325.0, phi=1.0, end_time=None):
    return SimulationSpec(
        sim_type=sim_type,
        fuel=fuel,
        oxidizer="O2:1.0, N2:3.76",
        phi=phi,
        T=T,
        P=P,
        end_time=end_time,
    )


def _try_codegen_with_retries(spec, max_retries=3):
    """Mirror the orchestrator's retry loop. Returns (success, result_dict, attempts)."""
    docs = retriever.retrieve(spec)
    prior_code, prior_error = None, None
    for attempt in range(1, max_retries + 1):
        code = codegen.run(spec, docs, prior_code, prior_error, attempt=attempt)
        exec_result = executor.run(code)
        if exec_result.success:
            return True, exec_result.result, attempt
        prior_code = code
        prior_error = exec_result.stderr
    return False, None, max_retries


# ============ Equilibrium ============

def test_codegen_equilibrium_ch4_air():
    """CH4/air at stoichiometric, 300 K, 1 atm → ~2226 K within 2%."""
    spec = _make_spec("equilibrium", fuel="CH4")
    ok, result, attempts = _try_codegen_with_retries(spec)
    assert ok, f"CodeGen failed after {attempts} attempts"
    assert "T_ad_K" in result, f"Missing T_ad_K in result: {result}"
    T = result["T_ad_K"]
    assert 2180 < T < 2280, f"Expected ~2226 K, got {T}"


def test_codegen_equilibrium_h2_air():
    """H2/air at stoichiometric, 298 K, 1 atm → ~2383 K within 2%."""
    spec = _make_spec("equilibrium", fuel="H2", T=298.0)
    ok, result, attempts = _try_codegen_with_retries(spec)
    assert ok, f"CodeGen failed after {attempts} attempts"
    T = result["T_ad_K"]
    assert 2330 < T < 2430, f"Expected ~2383 K, got {T}"


# ============ Ignition delay ============

def test_codegen_ignition_delay_h2():
    """H2/air at phi=1, 1200 K, 1 atm → reasonable ignition delay (1e-5 to 1e-3 s)."""
    spec = _make_spec(
        "ignition_delay", fuel="H2", T=1200.0, end_time=1.0
    )
    ok, result, attempts = _try_codegen_with_retries(spec)
    assert ok, f"CodeGen failed after {attempts} attempts"
    assert "ignition_delay_s" in result
    tau = result["ignition_delay_s"]
    assert 1e-5 < tau < 1e-3, f"Ignition delay {tau} out of expected range"


# ============ Flame speed ============

def test_codegen_flame_speed_ch4():
    """CH4/air at phi=1, 300 K, 1 atm → ~0.38 m/s within 15%.

    Flame speed is the hardest sim type: the 1D solver can fail to converge,
    or converge to a wrong solution. Wider tolerance (15%) acknowledges this.
    """
    spec = _make_spec("flame_speed", fuel="CH4")
    ok, result, attempts = _try_codegen_with_retries(spec)
    assert ok, f"CodeGen failed after {attempts} attempts"
    assert "flame_speed_m_s" in result
    s = result["flame_speed_m_s"]
    assert 0.32 < s < 0.45, f"Expected ~0.38 m/s, got {s}"
