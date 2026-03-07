"""Runtime patch for NeMo on Windows (signal.SIGKILL compatibility)."""

import logging
import signal
import sys

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_nemo_patch() -> bool:
    """Apply SIGKILL monkey-patch on Windows before NeMo is imported.

    NeMo's exp_manager.py references ``signal.SIGKILL`` which does not exist
    on Windows.  This adds the attribute (mapped to SIGTERM) so the import
    succeeds without modifying any installed files.

    Returns True if the patch was applied, False if it was unnecessary.
    """
    global _patch_applied

    if sys.platform != "win32":
        return False

    if hasattr(signal, "SIGKILL"):
        logger.debug("signal.SIGKILL already exists — NeMo patch not needed")
        return False

    if _patch_applied:
        return False

    signal.SIGKILL = signal.SIGTERM  # type: ignore[attr-defined]
    _patch_applied = True
    logger.info("Applied NeMo Windows compatibility patch (SIGKILL → SIGTERM)")
    return True
