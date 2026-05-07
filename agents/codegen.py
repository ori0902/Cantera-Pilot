"""CodeGen agent: SimulationSpec + retrieved docs → executable Python script.

Design philosophy (different from v1):
- System prompt carries only task contract (input format, output format,
  hard rules). No Cantera API rules, no gotchas list, no canonical template.
- Retrieved documentation is the primary signal for what API to call.
- On retry, temperature escalates so the model doesn't deterministically
  regenerate the same wrong output.

If this approach fails consistently on 8B, it's evidence that the 8B model
can't reliably write Cantera 3.2 code even when given the exact docstrings —
which would be a real finding, not a prompt failure.
"""
from __future__ import annotations
from typing import Optional
from llm_client import chat
from schemas import SimulationSpec
from config import MAX_TOKENS, TEMPERATURE


SYSTEM_PROMPT = """You are an expert Python programmer writing Cantera simulation \
scripts. You will be given:

1. A simulation specification (fuel, oxidizer, phi, T, P, etc.)
2. Relevant Cantera documentation and example scripts (use these as authoritative)

Produce a single self-contained Python script that runs the simulation and prints \
the numeric result as JSON.

SCRIPT CONTRACT (your script will be auto-checked):
1. Import only: cantera, numpy, math, json, sys. No other imports.
2. The final two lines must be:
     result = { ... }
     print(json.dumps(result))
3. All values in `result` must be JSON-serializable (float/int/str/list). \
Cast with float() where needed (numpy scalars are NOT json-serializable).
4. No interactive calls (no input(), plt.show(), etc.)

EXPECTED RESULT KEYS BY sim_type:
  equilibrium     →  {"T_ad_K": float, "P_Pa": float}
  ignition_delay  →  {"ignition_delay_s": float, "T_final_K": float}
  flame_speed     →  {"flame_speed_m_s": float}

CRITICAL — set_equivalence_ratio RULES (violations cause every run to fail):
  SIGNATURE: gas.set_equivalence_ratio(phi, fuel, oxidizer)
    - Three POSITIONAL arguments only. No keyword arguments whatsoever.
    - phi:     a plain float, e.g. 1.0
    - fuel:    a Cantera composition STRING, e.g. "CH4:1.0" or "c12h26:1.0"
               NEVER pass a bare species name like "CH4" — always append ":1.0"
    - oxidizer: a Cantera composition STRING, e.g. "O2:1.0, N2:3.76"
  CORRECT:   gas.set_equivalence_ratio(1.0, "c12h26:1.0", "O2:1.0, N2:3.76")
  WRONG:     gas.set_equivalence_ratio(phi=1.0, fuel="c12h26", oxidizer="O2:1.0, N2:3.76")
  WRONG:     gas.set_equivalence_ratio(1.0, "c12h26", "O2:1.0, N2:3.76")
  WRONG:     gas.set_equivalence_ratio(1.0, fuel="c12h26:1.0", reactants=...)

To build the fuel string from the spec variable: fuel_str = fuel + ":1.0"

REQUIRED API CALL SEQUENCE (these are mandatory — omitting any step produces wrong results):

equilibrium:
  gas = ct.Solution(mechanism)
  gas.TP = T, P                              # MUST set state before composition
  gas.set_equivalence_ratio(phi, fuel + ":1.0", oxidizer)
  gas.equilibrate('HP')                      # HP (not UV or TP) for adiabatic flame temp
  # then extract gas.T, gas.P

ignition_delay:
  gas = ct.Solution(mechanism)
  gas.TP = T, P
  gas.set_equivalence_ratio(phi, fuel + ":1.0", oxidizer)
  r = ct.IdealGasConstPressureReactor(gas, clone=False)
  net = ct.ReactorNet([r])
  # Integrate with net.advance() (NOT r.advance — the reactor has no advance method).
  # Step to a specific time using small uniform steps like 1e-5 s:
  #     times, temps = [], []
  #     T0 = gas.T
  #     while net.time < end_time:
  #         net.advance(net.time + 1e-5)
  #         times.append(net.time); temps.append(gas.T)
  #         if gas.T > T0 + 400: break
  # Ignition delay = time of peak dT/dt:
  #     dTdt = np.gradient(np.array(temps), np.array(times))
  #     tau = float(times[int(np.argmax(dTdt))])

flame_speed:
  gas = ct.Solution(mechanism)
  gas.TP = T, P
  gas.set_equivalence_ratio(phi, fuel + ":1.0", oxidizer)
  f = ct.FreeFlame(gas, width=0.03)
  f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
  f.solve(loglevel=0, auto=True)
  # flame_speed = f.velocity[0]

Use the provided documentation for exact function signatures and parameter details. \
The documentation reflects the exact installed Cantera version; prefer it over \
your training memory if they conflict.

Output ONLY the Python code. No markdown fences, no prose, no explanation."""


def _build_user_prompt(
    spec: SimulationSpec,
    retrieved_docs: str,
    prior_code: Optional[str] = None,
    prior_error: Optional[str] = None,
) -> str:
    """Assemble the user-turn prompt from spec, docs, and optional retry context.

    Args:
        spec:          Validated simulation specification.
        retrieved_docs: RAG-retrieved Cantera documentation string.
        prior_code:    The previously generated script (retry only).
        prior_error:   The stderr from the failed execution (retry only).

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    parts = []

    # Lead with retrieved docs when available — primary signal
    if retrieved_docs:
        parts.append("=== RELEVANT CANTERA DOCUMENTATION ===")
        parts.append(retrieved_docs)
        parts.append("")

    parts.append("=== SIMULATION SPECIFICATION ===")
    parts.append(spec.model_dump_json(indent=2))

    if prior_code and prior_error:
        parts.append("")
        parts.append("=== YOUR PREVIOUS ATTEMPT FAILED ===")
        parts.append("Error:")
        parts.append(prior_error[:2000])
        parts.append("")
        parts.append("Previous code:")
        parts.append(prior_code)
        parts.append("")
        parts.append(
            "Produce a CORRECTED script. Re-read the CRITICAL rules above carefully. "
            "Pay special attention to set_equivalence_ratio: three positional args, "
            "fuel must be 'SPECIES:1.0' format, NO keyword arguments."
        )

    return "\n".join(parts)


def _strip_fences(text: str) -> str:
    """Strip markdown code fences if the model adds them despite instructions.

    Args:
        text: Raw LLM output string.

    Returns:
        Clean Python source code string.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def run(
    spec: SimulationSpec,
    retrieved_docs: str = "",
    prior_code: Optional[str] = None,
    prior_error: Optional[str] = None,
    attempt: int = 1,
) -> str:
    """Generate a Cantera simulation script from a spec and optional retry context.

    Temperature schedule on retry:
      attempt 1: base (0.1) — near-deterministic
      attempt 2: 0.4       — genuine variation
      attempt 3: 0.7       — force different code paths

    Rationale: at T=0.1, identical prompts yield identical outputs. If the
    first attempt is wrong, a retry at the same temperature produces the same
    wrong code and accomplishes nothing.

    Args:
        spec:          Validated simulation specification.
        retrieved_docs: RAG-retrieved Cantera documentation string.
        prior_code:    The previously generated script (retry only).
        prior_error:   The stderr from the failed execution (retry only).
        attempt:       1-based attempt number; controls temperature schedule.

    Returns:
        Raw Python source code as a string.
    """
    user_prompt = _build_user_prompt(spec, retrieved_docs, prior_code, prior_error)
    base_temp = TEMPERATURE["codegen"]
    temp = min(base_temp + 0.3 * (attempt - 1), 0.8)
    raw = chat(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=MAX_TOKENS["codegen"],
        temperature=temp,
    )
    return _strip_fences(raw)
