# Cantera Multi-Agent LLM Tool (v2)

Natural-language interface to Cantera combustion simulations via a four-agent
pipeline: **Planner → CodeGen → Executor → Analysis**, with traceback-driven retry.

## Why v2

v1 went broad-first (built all four agents quickly, debugged integration issues
simultaneously). This worked to demonstrate the architecture but made debugging
painful. Three specific lessons drove the v2 rebuild:

1. **Verify API signatures against the installed library before writing prompts.**
   v1 had wrong rules in the system prompt (e.g., "pass `'air'` as-is") that
   actively caused failures. v2 starts from three hand-verified Cantera 3.2
   reference scripts in `reference/`, and the CodeGen prompt is built from
   those — not from guesswork.

2. **Deterministic transformations belong in Python, not prompts.**
   Oxidizer normalization (`'air'` → `'O2:1.0, N2:3.76'`), unit conversions,
   and sim-type classification are done in code before the LLM sees the spec.
   LLM prompts only handle genuinely LLM-shaped problems.

3. **Each layer passes tests before the next is built.** The executor runs all
   three reference scripts successfully before any agent is wired in. The
   Planner is tested against labeled queries before CodeGen calls it. Etc.

## Build order (gated by tests)

1. Sandbox + executor → **GATE**: reference scripts execute to correct values
2. Schemas + spec normalization → unit tests for the normalizer
3. RAG corpus build + index → smoke test: retrieval returns relevant chunks
4. Planner agent → tested against 10 labeled queries
5. CodeGen agent → tested per-sim-type on 3+ cases each
6. Validators → tested with deliberately-broken outputs
7. Analysis agent
8. Orchestrator + eval harness

## Current gate

```bash
source cantera_pilot/bin/activate
python -m pytest tests/test_executor.py -v
```

All tests must pass, including the three reference-script gate tests, before
we move to the next layer. If `test_reference_flame_speed_produces_correct_value`
takes 30-60 seconds that's normal — the 1D solver has to converge.
