"""Sandboxed script executor.

Defense in depth:
1. AST static check: parse the script, reject disallowed imports before running.
2. Subprocess isolation: fresh Python process, not eval() in-process.
3. Hard wall-clock timeout.
4. Structured output: script MUST print a single JSON line; we parse that.

This module has no dependencies on any agent — it's the pure execution layer
and is fully testable against the reference scripts in reference/.
"""
from __future__ import annotations
import ast
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from config import EXECUTION_TIMEOUT_SEC, ALLOWED_IMPORTS


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration_sec: float = 0.0
    result: Optional[dict] = None  # parsed from the last JSON line of stdout


class DisallowedImportError(Exception):
    pass


def _static_check(code: str) -> None:
    """Walk the AST and reject disallowed imports. Raises on violation.
    Syntax errors are NOT raised here — we let execution surface a real traceback
    the CodeGen agent can learn from on retry."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    raise DisallowedImportError(
                        f"Import of '{alias.name}' not allowed. "
                        f"Permitted: {sorted(ALLOWED_IMPORTS)}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    raise DisallowedImportError(
                        f"Import from '{node.module}' not allowed."
                    )


def _parse_last_json_line(stdout: str) -> Optional[dict]:
    """Script prints one JSON line at the end. Walk from the bottom so
    warnings/progress above don't confuse us."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run(code: str, workdir: Optional[Path] = None) -> ExecutionResult:
    """Execute code in a subprocess. Never raises; always returns ExecutionResult."""
    try:
        _static_check(code)
    except DisallowedImportError as e:
        return ExecutionResult(
            success=False,
            stderr=f"Static check failed: {e}",
            returncode=-1,
        )

    workdir = workdir or Path(tempfile.mkdtemp(prefix="cantera_sim_"))
    workdir.mkdir(parents=True, exist_ok=True)
    script_path = workdir / "sim.py"
    script_path.write_text(code)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SEC,
            cwd=str(workdir),
        )
        duration = time.perf_counter() - t0
        stdout = proc.stdout
        stderr = proc.stderr
        success = proc.returncode == 0
        result_dict = _parse_last_json_line(stdout) if success else None

        # Exit 0 with no JSON = failure, downstream needs a result dict
        if success and result_dict is None:
            success = False
            stderr = (stderr or "") + \
                "\n[executor] Script exited 0 but printed no JSON result line."

        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
            duration_sec=duration,
            result=result_dict,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            success=False,
            stdout=e.stdout.decode() if e.stdout else "",
            stderr=f"Timeout after {EXECUTION_TIMEOUT_SEC}s",
            returncode=-1,
            duration_sec=EXECUTION_TIMEOUT_SEC,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            stderr=f"Executor internal error: {type(e).__name__}: {e}",
            returncode=-1,
            duration_sec=time.perf_counter() - t0,
        )


def run_file(script_path: Path) -> ExecutionResult:
    """Convenience: read a file and execute it. Used by tests against reference scripts."""
    return run(script_path.read_text(), workdir=script_path.parent)
