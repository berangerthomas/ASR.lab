"""Automatic download and extraction of Vosk models."""

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

VOSK_BASE_URL = "https://alphacephei.com/vosk/models"


def _is_model_valid(model_dir: Path) -> bool:
    """Check whether a Vosk model directory looks complete.

    A valid model contains at least one of the expected structural markers
    (sub-directories or key files).  This catches empty dirs and partial
    extractions.
    """
    if not model_dir.is_dir():
        return False

    markers = ("am", "conf", "graph", "README", "ivector", "rnnlm")
    for entry in model_dir.iterdir():
        if entry.name in markers:
            return True
    return False


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "ASR.lab/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            disable=total == 0,
        ) as bar:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def ensure_vosk_model(model_path: str) -> bool:
    """Ensure the Vosk model at *model_path* is present and valid.

    If the directory does not exist or is incomplete the model ZIP is
    downloaded from alphacephei.com and extracted automatically.

    Returns True if a download was performed, False if the model was
    already available.
    """
    model_dir = Path(model_path)
    model_name = model_dir.name  # e.g. "vosk-model-en-us-0.22"
    parent_dir = model_dir.parent  # e.g. "models"

    # --- Already installed? ---
    if _is_model_valid(model_dir):
        logger.info("Vosk model %s: already installed ✓", model_name)
        return False

    # --- Sanity check: looks like a standard Vosk model name? ---
    if not model_name.startswith("vosk-model"):
        logger.warning(
            "Vosk model path '%s' does not follow the vosk-model-* naming "
            "convention — automatic download skipped. Please provide the "
            "model manually.",
            model_path,
        )
        return False

    url = f"{VOSK_BASE_URL}/{model_name}.zip"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary file next to the target so we stay on the same
    # filesystem (avoids cross-device rename issues).
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=parent_dir)
    tmp_file = Path(tmp_path)

    try:
        logger.info("Downloading Vosk model %s from %s …", model_name, url)
        _download_with_progress(url, tmp_file)

        # --- Validate ZIP ---
        if not zipfile.is_zipfile(tmp_file):
            raise RuntimeError(f"Downloaded file is not a valid ZIP: {url}")

        # --- Extract ---
        logger.info("Extracting %s …", model_name)
        with zipfile.ZipFile(tmp_file, "r") as zf:
            zf.extractall(parent_dir)

        # --- Verify extraction ---
        if not _is_model_valid(model_dir):
            raise RuntimeError(
                f"Extraction succeeded but model directory '{model_dir}' "
                "does not contain expected files."
            )

        logger.info("Vosk model %s installed successfully ✓", model_name)
        return True

    except Exception:
        # Clean up partial extraction on failure
        if model_dir.exists() and not _is_model_valid(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        raise

    finally:
        # Always clean up the ZIP
        tmp_file.unlink(missing_ok=True)
