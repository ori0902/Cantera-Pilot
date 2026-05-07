"""Planner agent: natural-language query → raw dict.

This agent is DELIBERATELY narrow. It does not:
  - Convert temperatures (it emits T_value + T_unit; normalize.py converts)
  - Expand oxidizer shorthand (it emits 'air'; normalize.py expands)
  - Map phi aliases (it emits 'stoichiometric'; normalize.py maps)
  - Do bounds validation (normalize_spec's Pydantic call does)

It ONLY identifies the fields and their units. Every downstream transformation
is deterministic Python. The LLM's job is pattern recognition, not arithmetic
or library knowledge — because v1 showed us that making the LLM do arithmetic
or remember unit rules is where bugs live.
"""
from __future__ import annotations
from llm_client import chat_json
from config import MAX_TOKENS, TEMPERATURE


SYSTEM_PROMPT = """You are a combustion simulation query parser. Given a natural-\
language query, extract the simulation fields as strict JSON.

Return exactly these fields (emit units SEPARATELY from values — do not convert):

{
  "sim_type":  one of "equilibrium", "ignition_delay", "flame_speed"
                - "adiabatic flame temperature" / "equilibrium" → "equilibrium"
                - "ignition delay" → "ignition_delay"
                - "laminar flame speed" / "flame speed" → "flame_speed"
  "fuel":      species symbol as used in mechanisms (e.g. "CH4", "H2", "C2H6", "C3H8")
  "oxidizer":  what the user said — "air", "O2", or an explicit composition string.
               Pass through as-is; DO NOT expand "air" yourself.
  "phi":       equivalence ratio. A number, OR one of the strings "stoichiometric",
               "lean", "rich". Pass through as-is; DO NOT convert "stoichiometric"
               to 1.0 yourself.
  "T_value":   temperature magnitude (number)
  "T_unit":    "K", "C", or "F" — exactly what the user said; DO NOT convert
  "P_value":   pressure magnitude (number)
  "P_unit":    "Pa", "kPa", "atm", "bar", "psi", "torr" — exactly what the user said
  "mechanism": optional; "gri30.yaml" if not specified
  "end_time":  optional; only for ignition_delay queries (seconds)
}

Examples:

Query: "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K"
JSON: {"sim_type":"equilibrium","fuel":"CH4","oxidizer":"air","phi":"stoichiometric","T_value":300,"T_unit":"K","P_value":1,"P_unit":"atm"}

Query: "ignition delay of H2/air at phi=0.5, 1200 K, 10 bar"
JSON: {"sim_type":"ignition_delay","fuel":"H2","oxidizer":"air","phi":0.5,"T_value":1200,"T_unit":"K","P_value":10,"P_unit":"bar"}

Query: "laminar flame speed of rich propane-air at 1 atm and 25 C"
JSON: {"sim_type":"flame_speed","fuel":"C3H8","oxidizer":"air","phi":"rich","T_value":25,"T_unit":"C","P_value":1,"P_unit":"atm"}

Query: "adiabatic flame temp of methane in pure oxygen at 298 K, 1 atm"
JSON: {"sim_type":"equilibrium","fuel":"CH4","oxidizer":"O2","phi":"stoichiometric","T_value":298,"T_unit":"K","P_value":1,"P_unit":"atm"}

CRITICAL: output ONLY the JSON object. No prose, no markdown fences, no explanation.
Do not transform values — the downstream code handles unit conversion and alias \
expansion. Your job is pattern recognition, not arithmetic."""


def run(query: str) -> dict:
    """Parse a natural-language query into a raw dict. Normalize downstream.

    Returns:
        Dict with the fields listed in SYSTEM_PROMPT. Ready to pass to
        normalize.normalize_spec().
    """
    return chat_json(
        system=SYSTEM_PROMPT,
        user=query,
        max_tokens=MAX_TOKENS["planner"],
        temperature=TEMPERATURE["planner"],
    )
