from __future__ import annotations

import os
from collections.abc import Mapping


def utf8_subprocess_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child-process environment with Python UTF-8 mode enabled.

    Inspect Evals currently contains resource readers that omit an explicit
    encoding. On Chinese Windows those readers inherit CP936/GBK and fail while
    importing the global task registry, before the requested task can start.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env
