"""Vocalis configuration helpers."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logger for configuration warnings
logger = logging.getLogger(__name__)

# Repository paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = Path(
    os.getenv("MODEL_CACHE_DIR", str(_REPO_ROOT / "models"))
).expanduser().resolve()


def _load_json_env(var_name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a JSON object stored in an environment variable."""

    raw_value = os.getenv(var_name)

    if not raw_value:
        return default

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        logger.warning("Unable to decode JSON for %s. Using default.", var_name)
        return default

    if not isinstance(parsed, dict):
        logger.warning(
            "Environment variable %s must be a JSON object. Using default.",
            var_name,
        )
        return default

    return parsed


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

AVAILABLE_STT_MODELS: List[Dict[str, Any]] = [
    {
        "id": "kyutai/stt-1b-en_fr",
        "label": "Kyutai STT 1B (English/French)",
        "description": "Low-latency bilingual streaming model optimised for conversational speech.",
        "generation_config": {"max_new_tokens": 256},
        "torch_dtype": "float16",
        "sample_rate": 16000,
    },
    {
        "id": "kyutai/stt-2.6b-en",
        "label": "Kyutai STT 2.6B (English)",
        "description": "Highest accuracy Kyutai release with slightly higher latency footprint.",
        "generation_config": {"max_new_tokens": 256},
        "torch_dtype": "float16",
        "sample_rate": 16000,
    },
]

AVAILABLE_TTS_MODELS: List[Dict[str, Any]] = [
    {
        "id": "sesame/csm-1b",
        "label": "Seseme CSM 1B",
        "description": "Codec-style high fidelity model (requires gated access).",
        "provider": "huggingface-local",
        "output_format": "wav",
    },
    {
        "id": "kyutai/tts-0.75b-en-public",
        "label": "Kyutai TTS 0.75B (English)",
        "description": "Entry level Kyutai TTS for lightweight hardware.",
        "provider": "huggingface-local",
        "output_format": "wav",
    },
    {
        "id": "kyutai/tts-1.6b-en_fr",
        "label": "Kyutai TTS 1.6B (English/French)",
        "description": "Mid-sized Kyutai TTS with bilingual support.",
        "provider": "huggingface-local",
        "output_format": "wav",
    },
]


def _find_model_option(options: List[Dict[str, Any]], model_id: str) -> Optional[Dict[str, Any]]:
    for option in options:
        if option.get("id") == model_id:
            return option
    return None


def get_available_models() -> Dict[str, Any]:
    """Return the configured model options."""

    return {
        "stt": AVAILABLE_STT_MODELS,
        "tts": AVAILABLE_TTS_MODELS,
    }


def get_stt_option(model_id: str) -> Optional[Dict[str, Any]]:
    return _find_model_option(AVAILABLE_STT_MODELS, model_id)


def get_tts_option(model_id: str) -> Optional[Dict[str, Any]]:
    return _find_model_option(AVAILABLE_TTS_MODELS, model_id)


# API Endpoints
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
TTS_API_ENDPOINT = os.getenv("TTS_API_ENDPOINT", "http://localhost:5005/v1/audio/speech")

# Speech-to-Text Configuration
STT_MODEL_ID = os.getenv(
    "STT_MODEL_ID",
    os.getenv("WHISPER_MODEL", "kyutai/stt-1b-en_fr"),
)
STT_DEVICE = os.getenv("STT_DEVICE")
STT_TORCH_DTYPE = os.getenv("STT_TORCH_DTYPE")
_DEFAULT_STT_OPTION = get_stt_option(os.getenv("WHISPER_MODEL", "kyutai/stt-1b-en_fr"))

STT_GENERATION_CONFIG = _load_json_env(
    "STT_GENERATION_CONFIG",
    (_DEFAULT_STT_OPTION or {}).get("generation_config", {"max_new_tokens": 256}),
)

# TTS Configuration
_DEFAULT_TTS_OPTION = get_tts_option(os.getenv("TTS_MODEL", "sesame/csm-1b"))

TTS_MODEL = os.getenv("TTS_MODEL", (_DEFAULT_TTS_OPTION or {}).get("id", "sesame/csm-1b"))
TTS_VOICE: Optional[str] = os.getenv("TTS_VOICE")
TTS_FORMAT = os.getenv("TTS_FORMAT", (_DEFAULT_TTS_OPTION or {}).get("output_format", "wav"))
TTS_PROVIDER = os.getenv("TTS_PROVIDER", (_DEFAULT_TTS_OPTION or {}).get("provider", "huggingface-local"))
TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_INFERENCE_PARAMS = _load_json_env("TTS_INFERENCE_PARAMS", {})
TTS_EXTRA_HEADERS = _load_json_env("TTS_EXTRA_HEADERS", {})

# WebSocket Server Configuration
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8000))

# Audio Processing
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.5))
VAD_BUFFER_SIZE = int(os.getenv("VAD_BUFFER_SIZE", 30))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))

def get_config() -> Dict[str, Any]:
    """
    Returns all configuration settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration settings
    """
    return {
        "llm_api_endpoint": LLM_API_ENDPOINT,
        "tts_api_endpoint": TTS_API_ENDPOINT,
        "stt_model_id": STT_MODEL_ID,
        "stt_device": STT_DEVICE,
        "stt_torch_dtype": STT_TORCH_DTYPE,
        "stt_generation_config": STT_GENERATION_CONFIG,
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "tts_format": TTS_FORMAT,
        "tts_provider": TTS_PROVIDER,
        "tts_api_key": TTS_API_KEY,
        "tts_inference_params": TTS_INFERENCE_PARAMS,
        "tts_extra_headers": TTS_EXTRA_HEADERS,
        "websocket_host": WEBSOCKET_HOST,
        "websocket_port": WEBSOCKET_PORT,
        "vad_threshold": VAD_THRESHOLD,
        "vad_buffer_size": VAD_BUFFER_SIZE,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "available_models": get_available_models(),
    }


def set_stt_model(model_id: str, generation_config: Optional[Dict[str, Any]] = None) -> None:
    """Persist the currently active STT model in module globals."""

    global STT_MODEL_ID, STT_GENERATION_CONFIG

    STT_MODEL_ID = model_id

    if generation_config is not None:
        STT_GENERATION_CONFIG = dict(generation_config)


def set_tts_model(
    model_id: str,
    *,
    provider: Optional[str] = None,
    voice: Optional[str] = None,
    output_format: Optional[str] = None,
    inference_params: Optional[Dict[str, Any]] = None,
    api_endpoint: Optional[str] = None,
) -> None:
    """Persist the currently active TTS model configuration."""

    global TTS_MODEL, TTS_PROVIDER, TTS_VOICE, TTS_FORMAT, TTS_INFERENCE_PARAMS, TTS_API_ENDPOINT

    TTS_MODEL = model_id

    if provider is not None:
        TTS_PROVIDER = provider

    if voice is not None:
        TTS_VOICE = voice

    if output_format is not None:
        TTS_FORMAT = output_format

    if inference_params is not None:
        TTS_INFERENCE_PARAMS = dict(inference_params)

    if api_endpoint is not None:
        TTS_API_ENDPOINT = api_endpoint
