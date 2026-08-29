"""Unit tests for utils/require_python.py.

The guard is a plain assertion over ``sys.version_info`` — the running
interpreter is 3.11 (nothing else is supported), so the pass path is
exercised as-is and the fail path is exercised by asking for a version
this interpreter is not.
"""

from __future__ import annotations

import sys

import pytest

from utils.require_python import REQUIRED, require_python


def test_current_interpreter_passes() -> None:
    # CI and every supported dev env run 3.11; this must be a silent no-op.
    require_python()


def test_required_tuple_is_311() -> None:
    assert REQUIRED == (3, 11)


def test_wrong_version_exits_nonzero_and_names_both_versions() -> None:
    wrong = (sys.version_info[0], sys.version_info[1] + 1)

    with pytest.raises(SystemExit) as excinfo:
        require_python(wrong)

    msg = str(excinfo.value)
    assert "requires Python 3." in msg
    assert f"{wrong[0]}.{wrong[1]}" in msg  # the version it wants
    assert f"{sys.version_info[0]}.{sys.version_info[1]}" in msg  # the one it found
    assert excinfo.value.code != 0
