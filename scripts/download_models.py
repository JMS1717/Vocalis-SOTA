#!/usr/bin/env python3
"""Download and cache Vocalis speech models from Hugging Face."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Dict, Any, Set

from huggingface_hub import snapshot_download

# Add the project root to the Python path to allow importing backend module
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import config  # noqa: E402

LOGGER = logging.getLogger("download_models")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _safe_directory_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _collect_models(include_stt: bool, include_tts: bool) -> Iterable[Dict[str, Any]]:
    if include_stt:
        yield from config.AVAILABLE_STT_MODELS
    if include_tts:
        yield from config.AVAILABLE_TTS_MODELS


def download_model(model_id: str, destination: Path) -> None:
    LOGGER.info("Ensuring model %s is available at %s", model_id, destination)

    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Kyutai/Seseme models for local inference")
    parser.add_argument(
        "--cache-dir",
        default=str(config.MODEL_CACHE_DIR),
        help="Directory where models should be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Download only STT models",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Download only TTS models",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Additional model repository IDs to download",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    include_stt = args.stt or not (args.stt or args.tts)
    include_tts = args.tts or not (args.stt or args.tts)

    models: Set[str] = set(args.model or [])

    for option in _collect_models(include_stt, include_tts):
        model_id = option.get("id")
        if model_id:
            models.add(model_id)

    if not models:
        LOGGER.warning("No models requested. Nothing to do.")
        return 0

    errors = False

    for model_id in sorted(models):
        target_dir = cache_dir / _safe_directory_name(model_id)
        try:
            download_model(model_id, target_dir)
        except Exception as exc:  # pragma: no cover - network/auth errors
            errors = True
            LOGGER.error("Failed to download %s: %s", model_id, exc)

    if errors:
        LOGGER.error("One or more models failed to download. Check authentication and retry.")
        return 1

    LOGGER.info("All requested models are available in %s", cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
