"""Evaluation test cases with reference values.

References are from Cantera 3.2 + GRI-Mech 3.0, verified directly on the user's
installed Cantera in the session that built this project. See reference/ for the
exact scripts used to establish these values.

Widened tolerances on ignition_delay because small changes in the operational
definition (peak dT/dt vs OH mass fraction peak vs T - T0 > 400) can shift the
value by factors of 1.5-3.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EvalCase:
    query: str
    expected_key: str
    expected_value: float
    tolerance_pct: float
    notes: str = ""


CASES: list[EvalCase] = [
    # ---- Equilibrium / adiabatic flame temperature ----
    EvalCase(
        query="adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K",
        expected_key="T_ad_K",
        expected_value=2226.0,
        tolerance_pct=2.0,
        notes="GRI-Mech 3.0, HP equilibration. Verified locally: 2225.52 K.",
    ),
    EvalCase(
        query="adiabatic flame temperature of H2/air at phi=1.0, 1 atm, 298 K",
        expected_key="T_ad_K",
        expected_value=2383.0,
        tolerance_pct=2.0,
    ),
    EvalCase(
        query="adiabatic flame temperature of lean methane-air (phi=0.7) at 1 atm, 300 K",
        expected_key="T_ad_K",
        expected_value=1839.0,
        tolerance_pct=3.0,
    ),
    EvalCase(
        query="adiabatic flame temp of rich CH4/air (phi=1.3) at 1 atm, 300 K",
        expected_key="T_ad_K",
        expected_value=2220.0,
        tolerance_pct=3.0,
    ),
    EvalCase(
        query="equilibrium temperature of propane-air at phi=1, 1 atm, 298 K",
        expected_key="T_ad_K",
        expected_value=2267.0,
        tolerance_pct=3.0,
    ),

    # ---- Ignition delay ----
    EvalCase(
        query="ignition delay of stoichiometric H2/air at 1200 K, 1 atm",
        expected_key="ignition_delay_s",
        expected_value=5e-5,
        tolerance_pct=200.0,
        notes="Verified locally at 0.0500 ms. Wide tolerance — "
              "operational definition of tau has significant variance.",
    ),
    EvalCase(
        query="ignition delay of stoichiometric H2/air at 1500 K, 1 atm",
        expected_key="ignition_delay_s",
        expected_value=1e-5,
        tolerance_pct=200.0,
    ),

    # ---- Flame speed ----
    EvalCase(
        query="laminar flame speed of stoichiometric CH4/air at 1 atm, 300 K",
        expected_key="flame_speed_m_s",
        expected_value=0.38,
        tolerance_pct=15.0,
        notes="Verified locally: 0.3851 m/s.",
    ),
    EvalCase(
        query="laminar flame speed of H2/air at phi=1, 1 atm, 298 K",
        expected_key="flame_speed_m_s",
        expected_value=2.10,
        tolerance_pct=20.0,
    ),
    EvalCase(
        query="flame speed of lean methane-air (phi=0.8) at 1 atm, 300 K",
        expected_key="flame_speed_m_s",
        expected_value=0.29,
        tolerance_pct=20.0,
    ),
]
