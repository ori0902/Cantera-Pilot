"""CLI entrypoint.

Usage:
    python run.py --query "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K"
    python run.py -q "ignition delay of H2/air at phi=1, 1200 K, 1 atm" --quiet
"""
from __future__ import annotations
import argparse
import sys
from orchestrator import run_query


def main() -> int:
    ap = argparse.ArgumentParser(description="Cantera multi-agent LLM tool")
    ap.add_argument("--query", "-q", required=True, help="Natural-language query")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    args = ap.parse_args()

    record = run_query(args.query, verbose=not args.quiet)

    print("\n" + "=" * 70)
    if record.execution_success and record.analysis:
        print("RESULT")
        print("=" * 70)
        print(record.analysis.summary)
        print("\nKey values:")
        for k, v in record.analysis.key_values.items():
            print(f"  {k}: {v}")
        if record.analysis.caveats:
            print("\nCaveats:")
            for c in record.analysis.caveats:
                print(f"  - {c}")
        print(f"\nCodeGen attempts: {record.codegen_attempts}")
        print(f"Total time: {record.total_duration_sec:.1f}s")
        return 0

    print("FAILED")
    print("=" * 70)
    print(record.error or "Unknown error")
    if record.execution_stderr:
        print(f"\nLast stderr (truncated):\n{record.execution_stderr[:500]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
