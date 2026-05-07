"""Orchestrator: runs Planner → normalize → retrieve → CodeGen (with retry) → Analysis.

Flow:
    query
      │
      ▼
    Planner (LLM, JSON mode) ──▶ raw dict
      │
      ▼
    normalize_spec (deterministic Python) ──▶ SimulationSpec
      │
      ▼
    retriever.retrieve(spec) ──▶ relevant Cantera docs
      │
      ▼
    ┌────── retry loop (up to MAX_CODEGEN_RETRIES) ───────────────┐
    │  CodeGen(spec, docs, prior_code?, prior_error?) → Python     │
    │                      │                                        │
    │                      ▼                                        │
    │  Executor.run(code) → ExecutionResult                         │
    │                      │                                        │
    │    exec failed?  ────┼── retry with traceback                 │
    │                      │                                        │
    │    validator fail? ──┼── retry with plausibility hint         │
    │                      │                                        │
    │    all good? ────────┴── break                                │
    └───────────────────────────────────────────────────────────────┘
      │
      ▼
    Analysis(spec, result) ──▶ AnalysisReport

Per-run artifacts saved to logs/<timestamp>/:
    record.json          — full audit trail
    sim_attemptN.py      — each generated script
"""
from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path

from agents import planner, codegen, analysis
from sandbox import executor
from rag import retriever
import normalize
import validators
from schemas import RunRecord
from config import MAX_CODEGEN_RETRIES, LOG_DIR


def run_query(query: str, verbose: bool = True) -> RunRecord:
    """End-to-end pipeline for one user query. Never raises — errors go into RunRecord.error."""
    t0 = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = LOG_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    record = RunRecord(query=query)

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t0:6.1f}s] {msg}")

    # ---- 1. Plan ----
    try:
        log(f"Planning: {query!r}")
        raw = planner.run(query)
        spec = normalize.normalize_spec(raw)
        record.spec = spec
        log(f"  → {spec.sim_type} | {spec.fuel}/{spec.oxidizer} | "
            f"phi={spec.phi} T={spec.T}K P={spec.P}Pa")
    except Exception as e:
        record.error = f"Planner/normalize failed: {type(e).__name__}: {e}"
        log(record.error)
        _save(record, run_dir)
        return record

    # ---- 2. Retrieve ----
    docs = retriever.retrieve(spec)
    if docs:
        log(f"Retrieved {len(docs)} chars of Cantera docs")
    else:
        log("No RAG docs available (index not built?)")

    # ---- 3. CodeGen + Execute + Validate loop ----
    prior_code: str | None = None
    prior_error: str | None = None
    exec_result = None

    for attempt in range(1, MAX_CODEGEN_RETRIES + 1):
        log(f"CodeGen attempt {attempt}/{MAX_CODEGEN_RETRIES}")
        try:
            code = codegen.run(spec, docs, prior_code, prior_error, attempt=attempt)
        except Exception as e:
            record.error = f"CodeGen failed: {type(e).__name__}: {e}"
            log(record.error)
            _save(record, run_dir)
            return record

        record.generated_code = code
        record.codegen_attempts = attempt
        (run_dir / f"sim_attempt{attempt}.py").write_text(code)

        log(f"Executing ({len(code)} chars)...")
        exec_result = executor.run(code, workdir=run_dir)

        if not exec_result.success:
            log(f"  ✗ Execution failed: {exec_result.stderr[:200]}")
            prior_code = code
            prior_error = exec_result.stderr
            continue

        # Execution succeeded — run plausibility validators
        issues = validators.validate(spec, exec_result.result or {})
        fail_issues = [i for i in issues if i.severity == "fail"]
        if fail_issues:
            log("  ⚠ Execution OK but validation failed:")
            for issue in fail_issues:
                log(f"      {issue.reason}")
            prior_code = code
            prior_error = validators.format_for_retry(fail_issues)
            continue

        # Warnings are logged but don't trigger retry
        for issue in issues:
            if issue.severity == "warn":
                log(f"  ⚠ warning: {issue.reason}")

        log(f"  ✓ Success in {exec_result.duration_sec:.1f}s → {exec_result.result}")
        break

    record.execution_success = bool(exec_result and exec_result.success)
    record.execution_stderr = exec_result.stderr if exec_result else ""
    record.execution_result = exec_result.result if exec_result else None

    # ---- 4. Analyze ----
    if record.execution_success and record.execution_result:
        try:
            log("Analyzing...")
            report = analysis.run(
                spec, record.execution_result,
                exec_result.duration_sec if exec_result else 0.0,
            )
            record.analysis = report
            log(f"  {report.summary}")
        except Exception as e:
            # Don't fail the whole run if Analysis has trouble; keep numeric result
            log(f"  Analysis error (non-fatal): {type(e).__name__}: {e}")
            record.analysis = analysis.failed_report(str(e))
    else:
        record.error = f"All {MAX_CODEGEN_RETRIES} code-generation attempts failed"
        record.analysis = analysis.failed_report(record.execution_stderr)
        log(record.error)

    record.total_duration_sec = time.perf_counter() - t0
    _save(record, run_dir)
    log(f"Total: {record.total_duration_sec:.1f}s | logs at {run_dir}")
    return record


def _save(record: RunRecord, run_dir: Path) -> None:
    (run_dir / "record.json").write_text(record.model_dump_json(indent=2))
