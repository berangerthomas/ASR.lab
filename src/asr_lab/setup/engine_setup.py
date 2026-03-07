"""Pre-flight setup for ASR engines that require external resources."""

import logging

from ..core.models import BenchmarkConfig
from .nemo_patch import apply_nemo_patch
from .vosk_setup import ensure_vosk_model

logger = logging.getLogger(__name__)


def ensure_engines_ready(config: BenchmarkConfig) -> None:
    """Inspect *config* and prepare any engine-specific prerequisites.

    * **NeMo** — applies the Windows SIGKILL compatibility patch (runtime
      monkey-patch, no file modification).
    * **Vosk** — downloads and extracts models that are missing from the
      configured ``model_path``.

    This function is idempotent and safe to call multiple times.
    """
    engine_configs = config.engines
    if not engine_configs:
        return

    # --- NeMo: runtime patch (must run before any nemo import) ---
    if "nemo" in engine_configs:
        enabled = any(c.enabled for c in engine_configs["nemo"])
        if enabled:
            apply_nemo_patch()

    # --- Vosk: auto-download models ---
    if "vosk" in engine_configs:
        for vosk_cfg in engine_configs["vosk"]:
            if not vosk_cfg.enabled:
                continue
            model_path = getattr(vosk_cfg, "model_path", None)
            if not model_path:
                logger.warning("Vosk engine '%s' has no model_path — skipping setup", vosk_cfg.id)
                continue
            try:
                ensure_vosk_model(model_path)
            except Exception as e:
                logger.error(
                    "Failed to setup Vosk model for '%s': %s. "
                    "The engine will likely fail at load time.",
                    vosk_cfg.id,
                    e,
                )
