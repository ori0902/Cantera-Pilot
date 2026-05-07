"""Validator tests.

Philosophy: feed the validator deliberately-wrong results, confirm it flags them.
If a validator doesn't catch its intended failure mode in tests, it's not a safety
net — it's decoration.
"""
import pytest

import validators
from schemas import SimulationSpec


def _spec(sim_type="equilibrium", fuel="CH4", P=101325.0):
    return SimulationSpec(
        sim_type=sim_type,
        fuel=fuel,
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T=300.0,
        P=P,
    )


# ============ Normal results pass cleanly ============

def test_valid_equilibrium_result_no_issues():
    issues = validators.validate(_spec(), {"T_ad_K": 2226.0, "P_Pa": 101325.0})
    assert issues == []


def test_valid_ignition_delay_no_issues():
    issues = validators.validate(
        _spec(sim_type="ignition_delay"),
        {"ignition_delay_s": 5e-5, "T_final_K": 2800.0},
    )
    assert issues == []


def test_valid_flame_speed_no_issues():
    issues = validators.validate(
        _spec(sim_type="flame_speed"),
        {"flame_speed_m_s": 0.385},
    )
    assert issues == []


# ============ Bad results get caught ============

def test_absurdly_low_flame_temp_caught():
    """The v1 failure: T_ad_K = 300 (input temp reflected back)."""
    issues = validators.validate(_spec(), {"T_ad_K": 300.0, "P_Pa": 101325.0})
    assert len(issues) >= 1
    assert any("T_ad_K" in i.field for i in issues)
    assert issues[0].severity == "fail"


def test_absurdly_low_flame_temp_7K_caught():
    """Earlier session bug: equilibration without setting TP gave T_ad_K = 7.46."""
    issues = validators.validate(_spec(), {"T_ad_K": 7.46, "P_Pa": 101325.0})
    assert any("T_ad_K" in i.field for i in issues)


def test_absurdly_high_flame_temp_caught():
    """Catch e.g. runaway simulations or unit-confusion errors."""
    issues = validators.validate(_spec(), {"T_ad_K": 50000.0, "P_Pa": 101325.0})
    assert any("T_ad_K" in i.field for i in issues)


def test_ignition_delay_too_short_caught():
    """Sub-nanosecond delays aren't physical for combustion."""
    issues = validators.validate(
        _spec(sim_type="ignition_delay"),
        {"ignition_delay_s": 1e-12, "T_final_K": 2800.0},
    )
    assert any("ignition_delay_s" in i.field for i in issues)


def test_ignition_delay_too_long_caught():
    issues = validators.validate(
        _spec(sim_type="ignition_delay"),
        {"ignition_delay_s": 1000.0, "T_final_K": 2800.0},
    )
    assert any("ignition_delay_s" in i.field for i in issues)


def test_negative_flame_speed_caught():
    issues = validators.validate(
        _spec(sim_type="flame_speed"),
        {"flame_speed_m_s": -0.5},
    )
    assert any("flame_speed_m_s" in i.field for i in issues)


def test_pressure_drift_warns_not_fails():
    """HP equilibration drift = warning, not failure. CodeGen shouldn't retry for this."""
    issues = validators.validate(
        _spec(),
        {"T_ad_K": 2226.0, "P_Pa": 150000.0},  # 48% drift
    )
    p_issues = [i for i in issues if i.field == "P_Pa"]
    assert len(p_issues) >= 1
    assert p_issues[0].severity == "warn"


# ============ format_for_retry builds useful hint strings ============

def test_format_for_retry_includes_reason():
    issues = validators.validate(_spec(), {"T_ad_K": 7.46, "P_Pa": 101325.0})
    msg = validators.format_for_retry(issues)
    assert "implausible" in msg.lower()
    assert "T_ad_K" in msg
    assert "7.46" in msg or "7.460" in msg
