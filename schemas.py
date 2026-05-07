"""Schemas shared across agents.

Every inter-agent message is a validated Pydantic model. Validation errors
surface loudly at boundaries rather than turning into silent downstream bugs.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


SimType = Literal["equilibrium", "ignition_delay", "flame_speed"]


class SimulationSpec(BaseModel):
    """The canonical representation of a simulation request.

    Produced by the Planner (from natural language) and/or normalized from a
    raw dict. Consumed by CodeGen (to generate a script) and by the validators
    (to sanity-check results).
    """

    sim_type: SimType
    fuel: str = Field(..., description="Species symbol, e.g. 'CH4', 'H2', 'C2H6'")
    # Post-normalization oxidizer is always an explicit composition string
    # like 'O2:1.0, N2:3.76' that Cantera.set_equivalence_ratio accepts directly.
    oxidizer: str = Field(..., description="Composition string for Cantera")
    phi: float = Field(..., gt=0, le=10, description="Equivalence ratio")
    T: float = Field(..., gt=0, lt=5000, description="Temperature in Kelvin")
    P: float = Field(..., gt=0, description="Pressure in Pascals")
    mechanism: str = Field(default="gri30.yaml")
    end_time: Optional[float] = Field(
        default=None, description="Ignition-delay sim end time, seconds"
    )

    @field_validator("mechanism")
    @classmethod
    def _mech_has_yaml(cls, v: str) -> str:
        if not v.endswith((".yaml", ".cti")):
            raise ValueError("mechanism must be a .yaml or .cti file")
        return v


class AnalysisReport(BaseModel):
    summary: str
    key_values: dict = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """Audit trail of one user query. Saved to logs/<timestamp>/record.json."""
    query: str
    spec: Optional[SimulationSpec] = None
    generated_code: Optional[str] = None
    codegen_attempts: int = 0
    execution_success: bool = False
    execution_stderr: str = ""
    execution_result: Optional[dict] = None
    analysis: Optional[AnalysisReport] = None
    total_duration_sec: float = 0.0
    error: Optional[str] = None
