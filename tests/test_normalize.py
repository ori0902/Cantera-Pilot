"""Normalizer tests.

These are the most important tests in the project because they pin down
every deterministic transformation the Planner agent will depend on. If these
pass, the LLM never has to reason about unit conversions or composition strings.
"""
import pytest
from pydantic import ValidationError

import normalize
from schemas import SimulationSpec


# ============ Oxidizer ============

def test_air_expands_to_composition_string():
    assert normalize.normalize_oxidizer("air") == "O2:1.0, N2:3.76"


def test_air_case_insensitive():
    assert normalize.normalize_oxidizer("Air") == "O2:1.0, N2:3.76"
    assert normalize.normalize_oxidizer("AIR") == "O2:1.0, N2:3.76"


def test_pure_oxygen_expands():
    assert normalize.normalize_oxidizer("O2") == "O2:1.0"
    assert normalize.normalize_oxidizer("oxygen") == "O2:1.0"


def test_explicit_composition_passthrough():
    """Already-valid Cantera composition strings should not be touched."""
    explicit = "O2:1.0, N2:3.76, Ar:0.04"
    assert normalize.normalize_oxidizer(explicit) == explicit


def test_unknown_oxidizer_passthrough():
    """Unknown shorthands pass through unchanged — let Cantera decide if it's valid."""
    assert normalize.normalize_oxidizer("exotic_mix") == "exotic_mix"


# ============ phi ============

def test_phi_numeric():
    assert normalize.normalize_phi(1.5) == 1.5
    assert normalize.normalize_phi(1) == 1.0


def test_phi_stoichiometric():
    assert normalize.normalize_phi("stoichiometric") == 1.0
    assert normalize.normalize_phi("stoich") == 1.0


def test_phi_lean_rich():
    assert normalize.normalize_phi("lean") == 0.7
    assert normalize.normalize_phi("rich") == 1.3


def test_phi_numeric_string():
    assert normalize.normalize_phi("0.85") == 0.85


def test_phi_bad_string_raises():
    with pytest.raises(ValueError):
        normalize.normalize_phi("sparkling")


# ============ Temperature ============

def test_temperature_kelvin_passthrough():
    """THE critical test — v1 had a bug here. '300 K' MUST stay 300.0."""
    assert normalize.normalize_temperature(300, "K") == 300.0
    assert normalize.normalize_temperature(300, "kelvin") == 300.0
    assert normalize.normalize_temperature(300, "") == 300.0  # no-unit default


def test_temperature_celsius():
    assert normalize.normalize_temperature(25, "C") == pytest.approx(298.15)
    assert normalize.normalize_temperature(0, "celsius") == pytest.approx(273.15)


def test_temperature_fahrenheit():
    assert normalize.normalize_temperature(32, "F") == pytest.approx(273.15, abs=0.01)


def test_temperature_degree_symbol():
    """'°C' should work like 'C'."""
    assert normalize.normalize_temperature(100, "°C") == pytest.approx(373.15)


def test_temperature_unknown_unit_raises():
    with pytest.raises(ValueError):
        normalize.normalize_temperature(100, "Rankine")


# ============ Pressure ============

def test_pressure_pa_passthrough():
    assert normalize.normalize_pressure(101325, "Pa") == 101325.0


def test_pressure_atm_to_pa():
    assert normalize.normalize_pressure(1, "atm") == 101325.0
    assert normalize.normalize_pressure(10, "atm") == pytest.approx(1_013_250)


def test_pressure_bar_to_pa():
    assert normalize.normalize_pressure(1, "bar") == 100_000.0
    assert normalize.normalize_pressure(10, "bar") == 1_000_000.0


def test_pressure_kpa_to_pa():
    assert normalize.normalize_pressure(101.325, "kPa") == pytest.approx(101325)


def test_pressure_psi_to_pa():
    assert normalize.normalize_pressure(14.696, "psi") == pytest.approx(101325, rel=0.001)


def test_pressure_unknown_unit_raises():
    with pytest.raises(ValueError):
        normalize.normalize_pressure(1, "gigaparsecs")


# ============ Full spec normalization (integration) ============

def test_normalize_spec_minimal():
    spec = normalize.normalize_spec({
        "sim_type": "equilibrium",
        "fuel": "CH4",
        "oxidizer": "air",
        "phi": "stoichiometric",
        "T": 300,
        "P": 101325,
    })
    assert spec.sim_type == "equilibrium"
    assert spec.fuel == "CH4"
    assert spec.oxidizer == "O2:1.0, N2:3.76"
    assert spec.phi == 1.0
    assert spec.T == 300.0
    assert spec.P == 101325.0
    assert spec.mechanism == "gri30.yaml"


def test_normalize_spec_celsius_and_atm():
    """The composite test: Planner gives Celsius temp + atm pressure; both convert."""
    spec = normalize.normalize_spec({
        "sim_type": "equilibrium",
        "fuel": "CH4",
        "oxidizer": "air",
        "phi": 1.0,
        "T_value": 25, "T_unit": "C",
        "P_value": 1, "P_unit": "atm",
    })
    assert spec.T == pytest.approx(298.15)
    assert spec.P == pytest.approx(101325)


def test_normalize_spec_ignition_delay():
    spec = normalize.normalize_spec({
        "sim_type": "ignition_delay",
        "fuel": "H2",
        "oxidizer": "air",
        "phi": 1.0,
        "T": 1200,
        "P": 101325,
        "end_time": 1.0,
    })
    assert spec.sim_type == "ignition_delay"
    assert spec.end_time == 1.0


def test_spec_rejects_invalid_sim_type():
    with pytest.raises(ValidationError):
        normalize.normalize_spec({
            "sim_type": "explosion",
            "fuel": "TNT",
            "oxidizer": "air",
            "phi": 1.0,
            "T": 300,
            "P": 101325,
        })


def test_spec_rejects_negative_phi():
    with pytest.raises(ValidationError):
        normalize.normalize_spec({
            "sim_type": "equilibrium",
            "fuel": "CH4",
            "oxidizer": "air",
            "phi": -1.0,
            "T": 300,
            "P": 101325,
        })


def test_spec_rejects_impossible_temperature():
    with pytest.raises(ValidationError):
        normalize.normalize_spec({
            "sim_type": "equilibrium",
            "fuel": "CH4",
            "oxidizer": "air",
            "phi": 1.0,
            "T": 10000,  # way beyond any real combustion condition
            "P": 101325,
        })
