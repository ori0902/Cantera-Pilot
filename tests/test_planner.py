"""Planner integration tests. Require Ollama running.

These pin down end-to-end behavior: given a real NL query, can the LLM + the
normalizer together produce the right SimulationSpec?

We test Planner output through the normalizer because that's how it will
actually be used. Catching "LLM returned something that can't be normalized"
here means we catch prompt drift immediately, not at eval time.
"""
import pytest

from agents import planner
import normalize


# ============ Sim type classification ============

def test_equilibrium_from_adiabatic_flame_temp_phrasing():
    raw = planner.run(
        "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K"
    )
    assert raw["sim_type"] == "equilibrium"


def test_ignition_delay_classification():
    raw = planner.run("ignition delay of H2/air at phi=1.0, 1200 K, 1 atm")
    assert raw["sim_type"] == "ignition_delay"


def test_flame_speed_classification():
    raw = planner.run("laminar flame speed of CH4/air at phi=1.0, 1 atm, 300 K")
    assert raw["sim_type"] == "flame_speed"


# ============ The v1 regression: Kelvin pass-through ============

def test_kelvin_stays_kelvin_through_normalizer():
    """v1 bug: '300 K' → planner returned 573.15.
    In v2 the planner emits T_value=300, T_unit='K'; normalize converts correctly."""
    raw = planner.run(
        "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.T == pytest.approx(300.0, abs=1.0), \
        f"Kelvin not preserved end-to-end: T={spec.T}"


# ============ Units through the full pipeline ============

def test_celsius_converts_to_kelvin():
    raw = planner.run(
        "adiabatic flame temperature of CH4/air, phi=1, 25 C, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.T == pytest.approx(298.15, abs=1.0)


def test_atm_converts_to_pa():
    raw = planner.run(
        "flame speed of CH4/air at phi=1, 300 K, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.P == pytest.approx(101325, abs=10)


def test_bar_converts_to_pa():
    raw = planner.run(
        "ignition delay of H2/air at phi=1, 1200 K, 10 bar"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.P == pytest.approx(1_000_000, abs=100)


# ============ phi aliases ============

def test_stoichiometric_becomes_phi_1():
    raw = planner.run(
        "adiabatic flame temp of stoichiometric CH4/air at 300 K, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.phi == pytest.approx(1.0)


def test_lean_becomes_phi_07():
    raw = planner.run(
        "adiabatic flame temp of lean CH4/air at 300 K, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.phi == pytest.approx(0.7)


def test_numeric_phi():
    raw = planner.run(
        "ignition delay of H2/air at phi=0.5, 1200 K, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.phi == pytest.approx(0.5)


# ============ Oxidizer pass-through + normalization ============

def test_air_expands_through_pipeline():
    """'air' in the query → Cantera composition string after normalize."""
    raw = planner.run(
        "adiabatic flame temperature of CH4/air at phi=1, 300 K, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.oxidizer == "O2:1.0, N2:3.76"


def test_pure_oxygen_recognized():
    raw = planner.run(
        "adiabatic flame temp of CH4 in pure oxygen at phi=1, 298 K, 1 atm"
    )
    spec = normalize.normalize_spec(raw)
    assert spec.oxidizer == "O2:1.0"


# ============ Fuel variety ============

def test_hydrogen_fuel():
    raw = planner.run("ignition delay of H2/air at phi=1, 1200 K, 1 atm")
    assert raw["fuel"] == "H2"


def test_propane_fuel():
    raw = planner.run("flame speed of propane in air at phi=1.2, 300 K, 1 atm")
    spec = normalize.normalize_spec(raw)
    assert spec.fuel == "C3H8"
