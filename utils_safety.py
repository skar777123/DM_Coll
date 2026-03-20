"""
utils_safety.py
───────────────
Safety helpers for fault-tolerant module loading.

Running an SIGILL-prone import (e.g. numpy with AVX on ARMv7) inside
the main process would kill the daemon. This helper runs the import in
a child process instead, so a crash is isolated and detectable.
"""

import logging
import subprocess
import sys

log = logging.getLogger(__name__)


def is_module_safe(module_name: str) -> bool:
    """
    Returns True only if the module can be imported without fatal errors.

    Runs the import inside a subprocess so that a SIGILL or other crash
    does NOT propagate to the parent process.

    :param module_name: Bare module name (e.g. ``'ultralytics'``).
    :returns: True if safe to import, False otherwise.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True
        if result.returncode == -4:   # SIGILL
            log.warning(
                "Module '%s' caused Illegal Instruction (SIGILL). Disabled.",
                module_name,
            )
        else:
            log.debug(
                "Module '%s' import failed (exit code %d).",
                module_name,
                result.returncode,
            )
        return False
    except subprocess.TimeoutExpired:
        log.warning("Module '%s' import timed out. Disabled.", module_name)
        return False
    except Exception as exc:
        log.debug("Error checking module '%s': %s", module_name, exc)
        return False
