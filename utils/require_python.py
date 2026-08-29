"""Runtime Python-version guard for host-run scripts.

Rationale — F-04 / F-07 in ``docs/uat/clean_clone_test_findings.md``:

* The dependency install fails on Python 3.14 (``pandasai==3.0.0`` has no
  compatible release) with a wall of ``Requires-Python`` exclusions that
  never names the interpreter version as the cause.
* During the clean-clone reproducibility test the shell fell out of the
  project venv three times — each time silently resolving ``python`` to
  system Python 3.14 — and produced four convincing but false defect
  reports (see §6, Retractions).

A one-line version assertion at the top of every host-run entrypoint would
have prevented all of them, so that is what this module is.

It imports only the standard library and must be called before any
third-party import, so the guard fires under the wrong interpreter instead
of a downstream ``ModuleNotFoundError`` or resolver stall.
"""
from __future__ import annotations

import sys

REQUIRED: tuple[int, int] = (3, 11)


def require_python(required: tuple[int, int] = REQUIRED) -> None:
    """Exit immediately unless the running interpreter is exactly ``required``.

    ``.python-version`` pins the same value for tooling that reads it; this
    is the runtime backstop for when it is not read (a stale shell, a
    system ``python``, an un-activated venv).
    """
    found = sys.version_info[:2]
    if found != required:
        want = ".".join(str(p) for p in required)
        got = ".".join(str(p) for p in found)
        script = sys.argv[0] or "this script"
        raise SystemExit(
            f"{script}: requires Python {want}, found {got}. "
            f"Activate the project venv (see the Requirements section of "
            f"README.md / setup.md) and re-run."
        )
