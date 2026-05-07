"""Run all eval cases, compute the four M5 metrics.

Outputs aligned with the proposal's evaluation plan:
  1. Execution success rate (target ≥80%)
  2. Numerical accuracy — within-tolerance rate
  3. End-to-end latency — p50 and p90
  4. Per-case breakdown for qualitative review

Usage: python -m eval.runner
"""
from __future__ import annotations
import statistics

from orchestrator import run_query
from eval.cases import CASES


def main() -> None:
    rows: list[dict] = []
    for i, case in enumerate(CASES, 1):
        print(f"\n{'='*70}\n[{i}/{len(CASES)}] {case.query}\n{'='*70}")
        record = run_query(case.query, verbose=True)

        success = bool(record.execution_success and record.analysis)
        actual: float | None = None
        rel_err: float | None = None
        within_tol = False

        if success and record.execution_result:
            value = record.execution_result.get(case.expected_key)
            if isinstance(value, (int, float)):
                actual = float(value)
                rel_err = abs(actual - case.expected_value) / abs(case.expected_value) * 100
                within_tol = rel_err <= case.tolerance_pct

        rows.append({
            "query": case.query,
            "success": success,
            "latency_s": record.total_duration_sec,
            "attempts": record.codegen_attempts,
            "expected": case.expected_value,
            "actual": actual,
            "rel_err_pct": rel_err,
            "within_tol": within_tol,
            "tolerance_pct": case.tolerance_pct,
            "sim_type": record.spec.sim_type if record.spec else "?",
        })

    # ---- Aggregate ----
    n = len(rows)
    n_success = sum(r["success"] for r in rows)
    n_accurate = sum(r["within_tol"] for r in rows)
    latencies = sorted(r["latency_s"] for r in rows)
    p50 = statistics.median(latencies) if latencies else 0.0
    p90 = latencies[int(0.9 * (n - 1))] if n > 1 else (latencies[0] if latencies else 0.0)

    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"Cases:             {n}")
    print(f"Execution success: {n_success}/{n} = {n_success/n*100:5.1f}%  (target ≥80%)")
    print(f"Within tolerance:  {n_accurate}/{n} = {n_accurate/n*100:5.1f}%")
    print(f"Latency p50:       {p50:5.1f}s  (target <90s)")
    print(f"Latency p90:       {p90:5.1f}s  (target <120s)")

    # Per-sim-type breakdown helps spot which categories are strong/weak
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["sim_type"], []).append(r)
    print("\nBy sim_type:")
    for st, rs in sorted(by_type.items()):
        s = sum(r["success"] for r in rs)
        a = sum(r["within_tol"] for r in rs)
        print(f"  {st:16s} success {s}/{len(rs)}   within-tol {a}/{len(rs)}")

    print("\nPer-case:")
    for r in rows:
        mark = "✓" if r["within_tol"] else ("~" if r["success"] else "✗")
        err_str = f"{r['rel_err_pct']:5.1f}%" if r["rel_err_pct"] is not None else "   n/a"
        actual_str = f"{r['actual']:.3g}" if r["actual"] is not None else "  n/a"
        print(f"  {mark} [{r['attempts']}a] {r['latency_s']:5.1f}s  "
              f"exp={r['expected']:.3g}  act={actual_str}  err={err_str}  "
              f"{r['query'][:50]}")


if __name__ == "__main__":
    main()
