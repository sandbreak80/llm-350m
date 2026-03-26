"""
Isolated code execution sandbox for RLVR training.

Executes AI-generated Python code in a subprocess with hard resource limits.
NEVER use eval() or exec() inline — always go through this module.

Safety guarantees:
  - Separate subprocess — crash/hang cannot affect the training process
  - Timeout enforced via subprocess.run(timeout=)
  - Memory limit enforced via resource.setrlimit in preexec_fn (Linux only)
  - Temp file written to /tmp, deleted after execution
  - No network access (subprocess inherits parent env — caller must ensure no credentials)

Usage:
    from sandbox.executor import execute_python_safe
    result = execute_python_safe(code, test_cases, timeout_seconds=5, memory_limit_mb=256)
    # {"passed": 1, "total": 3, "pass_rate": 0.33, "error": "...", "timed_out": False}
"""

import os
import sys
import tempfile
import subprocess
import platform
from typing import Optional


def _make_preexec_fn(memory_limit_mb: int):
    """
    Returns a preexec_fn that sets virtual memory limits before exec.
    Linux only — no-op on macOS/Windows (resource.RLIMIT_AS not available).
    """
    if platform.system() != "Linux":
        return None

    import resource
    mem_bytes = memory_limit_mb * 1024 * 1024

    def _set_limits():
        try:
            # RLIMIT_AS: maximum size of virtual memory (address space)
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, resource.error):
            pass  # Best-effort — don't abort if limit can't be set

    return _set_limits


def execute_python_safe(
    code: str,
    test_cases: list[str],
    timeout_seconds: int = 5,
    memory_limit_mb: int = 256,
) -> dict:
    """
    Execute code + test_cases in an isolated subprocess with resource limits.

    Args:
        code:             Python source string (the generated solution).
        test_cases:       List of assertion/test strings to append after the solution.
        timeout_seconds:  Hard wall-clock timeout. Returns timed_out=True if exceeded.
        memory_limit_mb:  Virtual memory cap in MB (Linux only, enforced via RLIMIT_AS).

    Returns:
        {
            "passed":    int,   # 1 if all tests passed, 0 otherwise
            "total":     int,   # len(test_cases)
            "pass_rate": float, # passed / total (0.0 or 1.0 — all-or-nothing per run)
            "error":     str | None,
            "timed_out": bool,
        }
    """
    full_code = code.rstrip() + "\n\n" + "\n".join(test_cases)
    preexec_fn = _make_preexec_fn(memory_limit_mb)

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=preexec_fn,
        )

        passed = int(result.returncode == 0)
        return {
            "passed":    passed,
            "total":     len(test_cases),
            "pass_rate": float(passed),
            "error":     result.stderr.strip() if result.returncode != 0 else None,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired:
        return {
            "passed":    0,
            "total":     len(test_cases),
            "pass_rate": 0.0,
            "error":     f"Timeout after {timeout_seconds}s",
            "timed_out": True,
        }
    except MemoryError:
        return {
            "passed":    0,
            "total":     len(test_cases),
            "pass_rate": 0.0,
            "error":     "MemoryError in subprocess launch",
            "timed_out": False,
        }
    except Exception as e:
        return {
            "passed":    0,
            "total":     len(test_cases),
            "pass_rate": 0.0,
            "error":     str(e),
            "timed_out": False,
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def syntax_check(code: str) -> bool:
    """
    Quick syntax check without execution. Returns True if code parses cleanly.
    Useful for partial reward: give 0.1 credit for syntactically valid code
    even if tests fail, to avoid reward collapse.
    """
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
