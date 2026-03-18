
import sys
import subprocess
import logging

log = logging.getLogger(__name__)

def is_module_safe(module_name: str) -> bool:
    """
    Checks if a module can be imported without causing a SIGILL or other fatal errors.
    Runs the import in a subprocess to protect the main process.
    """
    try:
        # Run a simple import command in a separate process
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            timeout=15
        )
        if result.returncode == 0:
            return True
        if result.returncode == -4: # SIGILL
            log.warning(f"Module '{module_name}' caused Illegal Instruction (SIGILL). It will be disabled.")
        else:
            log.debug(f"Module '{module_name}' import failed with return code {result.returncode}")
        return False
    except subprocess.TimeoutExpired:
        log.warning(f"Module '{module_name}' import timed out. It will be disabled.")
        return False
    except Exception as e:
        log.debug(f"Error checking module '{module_name}': {e}")
        return False
