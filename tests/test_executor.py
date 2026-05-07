"""Executor tests.

Two categories:
1. Unit tests for the sandbox (no Cantera needed)
2. GATE TESTS: run the three reference scripts through the sandbox and confirm
   they produce the values we verified interactively on Cantera 3.2.

The gate tests exist so that before we write a single agent, we prove that
the executor can run correct Cantera code and extract correct numeric results.
If any gate test fails, nothing downstream can work. Do not skip.
"""
import pytest
from pathlib import Path

from sandbox import executor
from config import REFERENCE_DIR


# ============ Unit tests: sandbox mechanics ============

def test_simple_success():
    code = 'import json\nresult = {"x": 42}\nprint(json.dumps(result))'
    out = executor.run(code)
    assert out.success
    assert out.result == {"x": 42}


def test_blocks_disallowed_import():
    code = 'import os\nprint(os.getcwd())'
    out = executor.run(code)
    assert not out.success
    assert "not allowed" in out.stderr.lower()


def test_blocks_from_import():
    code = 'from subprocess import run\nprint("hi")'
    out = executor.run(code)
    assert not out.success
    assert "not allowed" in out.stderr.lower()


def test_runtime_error_surfaces_traceback():
    code = 'import json\nraise ValueError("kaboom")'
    out = executor.run(code)
    assert not out.success
    assert "ValueError" in out.stderr
    assert "kaboom" in out.stderr


def test_no_json_output_is_failure():
    code = 'print("ran but forgot to emit JSON")'
    out = executor.run(code)
    assert not out.success
    assert "no JSON" in out.stderr


def test_json_on_last_line_among_noise():
    """Real scripts often print warnings; we pick the last JSON line."""
    code = '''
import json
print("some progress log")
print("more logging")
print(json.dumps({"T_ad_K": 2225.52}))
'''
    out = executor.run(code)
    assert out.success
    assert out.result == {"T_ad_K": 2225.52}


# ============ GATE TESTS: reference scripts ============

def test_reference_equilibrium_produces_correct_temp():
    """GATE: must pass before any agent is built.
    CH4/air adiabatic flame temp should be ~2226 K within 2%."""
    script = REFERENCE_DIR / "equilibrium_ch4_air.py"
    out = executor.run_file(script)
    assert out.success, f"Reference script failed: {out.stderr}"
    assert "T_ad_K" in out.result
    T = out.result["T_ad_K"]
    assert 2180 < T < 2280, f"Expected ~2226 K, got {T}"


def test_reference_ignition_delay_produces_reasonable_value():
    """GATE: H2/air at 1200 K should ignite in 1e-5 to 1e-3 s."""
    script = REFERENCE_DIR / "ignition_delay_h2_air.py"
    out = executor.run_file(script)
    assert out.success, f"Reference script failed: {out.stderr}"
    assert "ignition_delay_s" in out.result
    tau = out.result["ignition_delay_s"]
    assert 1e-5 < tau < 1e-3, f"Expected 1e-5 to 1e-3 s, got {tau}"


def test_reference_flame_speed_produces_correct_value():
    """GATE: CH4/air flame speed should be ~0.38 m/s within 10%.
    This test takes longer (~30-60s) because the 1D solver has to converge."""
    script = REFERENCE_DIR / "flame_speed_ch4_air.py"
    out = executor.run_file(script)
    assert out.success, f"Reference script failed: {out.stderr}"
    assert "flame_speed_m_s" in out.result
    s = out.result["flame_speed_m_s"]
    assert 0.34 < s < 0.42, f"Expected ~0.38 m/s, got {s}"
