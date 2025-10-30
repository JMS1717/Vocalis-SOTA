"""Vocalis configuration helpers."""

import json
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logger for configuration warnings
logger = logging.getLogger(__name__)


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
STT_GENERATION_CONFIG = _load_json_env(
    "STT_GENERATION_CONFIG",
    {"max_new_tokens": 256},
)

# TTS Configuration
TTS_MODEL = os.getenv("TTS_MODEL", "sesame/csm-1b")
TTS_VOICE: Optional[str] = os.getenv("TTS_VOICE")
TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "huggingface-local")
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
    }
