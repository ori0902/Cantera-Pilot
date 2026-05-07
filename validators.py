"""Physical-plausibility validators for simulation results.

Tight enough to catch order-of-magnitude wrongness (the silent failures our
agent retry can't catch on its own), loose enough that any legitimate case
with phi in [0.3, 2.0] passes. These are safety net bounds, not a substitute
for the eval harness's accuracy checks.

Philosophy: bounds encode "is this even a combustion result?" not
"is this the exact right answer?" — the eval harness compares against
reference values for that.
"""
from __future__ import annotations
from dataclasses import dataclass
from schemas import SimulationSpec


@dataclass
class ValidationIssue:
    field: str
    value: float
    reason: str
    severity: str  # "warn" or "fail"


# Physical bounds by result-key. Hydrocarbon/H2 combustion range, phi ∈ [0.3, 2.0].
BOUNDS = {
    # Adiabatic flame temperature (HP). H2 stoich ~2380, CH4 stoich ~2226.
    # Very lean floor ~1500, very rich ~2500. Widen to catch real errors.
    "T_ad_K": (1000.0, 3500.0),
    # Ignition delay from sub-microsecond (high-T H2) to multiple seconds (very lean).
    "ignition_delay_s": (1e-9, 10.0),
    # Laminar flame speed: slowest (lean heavy HC) ~0.05, fastest (rich H2) ~3.5.
    "flame_speed_m_s": (0.05, 3.5),
    # Final temperature after ignition should be well above initial T.
    # Bound loosely; Analysis agent will flag absurd cases.
    "T_final_K": (500.0, 4000.0),
}


def validate(spec: SimulationSpec, result: dict) -> list[ValidationIssue]:
    """Return validation issues. Empty list = no problems."""
    issues: list[ValidationIssue] = []
    if not result:
        return issues

    for field, value in result.items():
        if field not in BOUNDS:
            continue
        if not isinstance(value, (int, float)):
            continue
        lo, hi = BOUNDS[field]
        if value < lo or value > hi:
            issues.append(ValidationIssue(
                field=field,
                value=float(value),
                reason=f"{field}={value:.4g} outside physical bounds [{lo}, {hi}]",
                severity="fail",
            ))

    # Cross-check: HP equilibration should preserve pressure within 1%
    if spec.sim_type == "equilibrium" and "P_Pa" in result:
        p_out = result["P_Pa"]
        if isinstance(p_out, (int, float)) and spec.P > 0:
            if abs(p_out - spec.P) / spec.P > 0.01:
                issues.append(ValidationIssue(
                    field="P_Pa",
                    value=float(p_out),
                    reason=f"Pressure drift {spec.P}→{p_out} Pa (HP should preserve P)",
                    severity="warn",
                ))

    return issues


def format_for_retry(issues: list[ValidationIssue]) -> str:
    """Build a hint string to feed back to CodeGen on validation failure."""
    lines = ["Your previous code ran successfully but produced physically implausible results:"]
    for issue in issues:
        lines.append(f"  - {issue.reason}")
    lines.append(
        "Likely causes: wrong equilibrate mode (use 'HP' for adiabatic flame temp), "
        "wrong units, wrong sign, or using the wrong formula entirely. "
        "Re-read the documentation and produce corrected code."
    )
    return "\n".join(lines)
