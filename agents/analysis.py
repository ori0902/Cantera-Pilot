"""Analysis agent: ExecutionResult → AnalysisReport.

Takes validated numeric outputs from the executor and produces:
- A 2-3 sentence plain-English summary
- key_values: pass-through of the numeric result dict
- caveats: mechanism applicability notes, assumptions

Simpler than Planner or CodeGen because there's no parsing risk — if
execution.result is a valid dict, we just interpret it.
"""
from __future__ import annotations
from llm_client import chat_json
from schemas import SimulationSpec, AnalysisReport
from config import MAX_TOKENS, TEMPERATURE


SYSTEM_PROMPT = """You are a combustion scientist explaining simulation results to \
a researcher. Given a simulation spec and its numeric outputs, return JSON:

{
  "summary": "2-3 sentences. Include the headline number and its physical meaning.",
  "key_values": { ...pass the numeric result dict through ... },
  "caveats": ["bullet-style notes like mechanism applicability or assumptions"]
}

Be precise with units (K, m/s, s, Pa). Flag suspicious values: hydrocarbon adiabatic \
flame temperature below 1500 K or above 3000 K, ignition delay below 1 ns or above \
10 s, flame speed below 0.05 m/s or above 3.5 m/s.

Output ONLY JSON, no prose or markdown."""


def run(spec: SimulationSpec, result: dict, duration_sec: float = 0.0) -> AnalysisReport:
    """Interpret numeric results. `result` must already be a valid dict (no error path)."""
    user_prompt = (
        f"Spec:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Numeric results:\n{result}\n\n"
        f"Execution time: {duration_sec:.1f} s."
    )
    raw = chat_json(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=MAX_TOKENS["analysis"],
        temperature=TEMPERATURE["analysis"],
    )
    return AnalysisReport(**raw)


def failed_report(execution_stderr: str) -> AnalysisReport:
    """Fallback when execution failed — no LLM call needed."""
    return AnalysisReport(
        summary=f"Simulation failed to complete. Error: {execution_stderr[:300]}",
        key_values={},
        caveats=["No results produced; see logs for full traceback."],
    )
