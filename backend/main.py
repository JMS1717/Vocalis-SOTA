"""
Vocalis Backend Server

FastAPI application entry point.
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Import configuration
from . import config

# Import services
from .services.transcription import SpeechTranscriber
from .services.llm import LLMClient
from .services.tts import TTSClient
from .services.vision import vision_service

# Import routes
from .routes.websocket import websocket_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global service instances
transcription_service = None
llm_service = None
tts_service = None
service_reload_lock = asyncio.Lock()
# Vision service is a singleton already initialized in its module


class ModelSelection(BaseModel):
    stt_model_id: Optional[str] = None
    tts_model_id: Optional[str] = None
    stt_generation_config: Optional[Dict[str, Any]] = None
    tts_inference_params: Optional[Dict[str, Any]] = None
    tts_voice: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI application.
    """
    # Load configuration
    cfg = config.get_config()
    
    # Initialize services on startup
    logger.info("Initializing services...")
    
    global transcription_service, llm_service, tts_service
    
    # Initialize transcription service
    transcription_service = SpeechTranscriber(
        model_id=cfg["stt_model_id"],
        device=cfg.get("stt_device"),
        torch_dtype=cfg.get("stt_torch_dtype"),
        sample_rate=cfg["audio_sample_rate"],
        generation_config=cfg["stt_generation_config"],
        cache_dir=str(config.MODEL_CACHE_DIR),
    )
    
    # Initialize LLM service
    llm_service = LLMClient(
        api_endpoint=cfg["llm_api_endpoint"]
    )
    
    # Initialize TTS service
    tts_service = TTSClient(
        api_endpoint=cfg["tts_api_endpoint"],
        model=cfg["tts_model"],
        voice=cfg["tts_voice"],
        output_format=cfg["tts_format"],
        provider=cfg["tts_provider"],
        api_key=cfg["tts_api_key"],
        inference_params=cfg["tts_inference_params"],
        extra_headers=cfg["tts_extra_headers"],
    )
    
    # Initialize vision service (will download model if not cached)
    logger.info("Initializing vision service...")
    vision_service.initialize()
    
    logger.info("All services initialized successfully")
    
    try:
        yield
    finally:
        logger.info("Shutting down services...")

        if tts_service:
            await tts_service.aclose()
            tts_service = None

        if llm_service:
            await llm_service.aclose()
            llm_service = None

        if transcription_service:
            transcription_service.close()
            transcription_service = None

        logger.info("Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Vocalis Backend",
    description="Speech-to-Speech AI Assistant Backend",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service dependency functions
def get_transcription_service():
    return transcription_service

def get_llm_service():
    return llm_service

def get_tts_service():
    return tts_service

# API routes
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "Vocalis backend is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "transcription": transcription_service is not None,
            "llm": llm_service is not None,
            "tts": tts_service is not None,
            "vision": vision_service.is_ready()
        },
        "config": {
            "stt_model_id": config.STT_MODEL_ID,
            "tts_voice": config.TTS_VOICE,
            "websocket_port": config.WEBSOCKET_PORT
        }
    }

@app.get("/config")
async def get_full_config():
    """Get full configuration."""
    if not all([transcription_service, llm_service, tts_service]) or not vision_service.is_ready():
        raise HTTPException(status_code=503, detail="Services not initialized")

    return {
        "transcription": transcription_service.get_config(),
        "llm": llm_service.get_config(),
        "tts": tts_service.get_config(),
        "system": config.get_config()
    }


@app.get("/models")
async def list_models():
    """Return available model options and current selections."""

    return {
        "stt": {
            "current": config.STT_MODEL_ID,
            "options": config.AVAILABLE_STT_MODELS,
        },
        "tts": {
            "current": config.TTS_MODEL,
            "options": config.AVAILABLE_TTS_MODELS,
        },
    }


@app.post("/models/select", status_code=status.HTTP_202_ACCEPTED)
async def select_models(selection: ModelSelection):
    """Reload STT and/or TTS services with the requested model identifiers."""

    global transcription_service, tts_service

    updates: Dict[str, Dict[str, Any]] = {}

    async with service_reload_lock:
        if selection.stt_model_id:
            option = config.get_stt_option(selection.stt_model_id)
            if option is None:
                raise HTTPException(status_code=400, detail="Unknown STT model")

            raw_generation_config = selection.stt_generation_config or option.get(
                "generation_config",
                config.STT_GENERATION_CONFIG,
            )
            generation_config = dict(raw_generation_config or {})

            dtype_override = generation_config.pop("torch_dtype", None)

            device = option.get("device", config.STT_DEVICE)
            torch_dtype = dtype_override or option.get("torch_dtype", config.STT_TORCH_DTYPE)

            current_config = transcription_service.get_config() if transcription_service else {}
            is_same_model = current_config.get("model_id") == option["id"]
            is_same_generation = generation_config == config.STT_GENERATION_CONFIG

            if is_same_model and is_same_generation:
                logger.info("Requested STT model %s is already active; skipping reload", option["id"])
            else:
                old_transcriber = transcription_service
                transcription_service = SpeechTranscriber(
                    model_id=option["id"],
                    device=device,
                    torch_dtype=torch_dtype,
                    sample_rate=option.get("sample_rate", config.AUDIO_SAMPLE_RATE),
                    generation_config=generation_config,
                    cache_dir=str(config.MODEL_CACHE_DIR),
                )

                config.set_stt_model(option["id"], generation_config)

                updates["stt"] = transcription_service.get_config()

                if old_transcriber is not None:
                    old_transcriber.close()

        if selection.tts_model_id:
            option = config.get_tts_option(selection.tts_model_id)
            if option is None:
                raise HTTPException(status_code=400, detail="Unknown TTS model")

            inference_params = selection.tts_inference_params or option.get(
                "inference_params",
                config.TTS_INFERENCE_PARAMS,
            )

            provider = option.get("provider", config.TTS_PROVIDER)
            output_format = option.get("output_format", config.TTS_FORMAT)
            voice = selection.tts_voice or option.get("voice", config.TTS_VOICE)

            api_endpoint = option.get("api_endpoint", config.TTS_API_ENDPOINT)
            current_tts_config = tts_service.get_config() if tts_service else {}
            is_same_tts_model = current_tts_config.get("model") == option["id"]
            is_same_voice = voice == config.TTS_VOICE
            is_same_provider = provider == config.TTS_PROVIDER
            is_same_format = output_format == config.TTS_FORMAT
            is_same_endpoint = api_endpoint == config.TTS_API_ENDPOINT
            is_same_params = inference_params == config.TTS_INFERENCE_PARAMS

            if all([is_same_tts_model, is_same_voice, is_same_provider, is_same_format, is_same_endpoint, is_same_params]):
                logger.info("Requested TTS model %s is already active; skipping reload", option["id"])
            else:
                old_tts = tts_service
                tts_service = TTSClient(
                    api_endpoint=api_endpoint,
                    model=option["id"],
                    voice=voice,
                    output_format=output_format,
                    provider=provider,
                    api_key=config.TTS_API_KEY,
                    inference_params=inference_params,
                    extra_headers=config.TTS_EXTRA_HEADERS,
                )

                config.set_tts_model(
                    option["id"],
                    provider=provider,
                    voice=voice,
                    output_format=output_format,
                    inference_params=inference_params,
                    api_endpoint=api_endpoint,
                )

                updates["tts"] = tts_service.get_config()

                if old_tts is not None:
                    await old_tts.aclose()

    if not updates:
        raise HTTPException(status_code=400, detail="No model changes requested")

    return {"status": "accepted", "updated": updates}

# WebSocket route
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for bidirectional audio streaming."""
    await websocket_endpoint(
        websocket, 
        transcription_service, 
        llm_service, 
        tts_service
    )

# Run server directly if executed as script
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=config.WEBSOCKET_HOST,
        port=config.WEBSOCKET_PORT,
        reload=True
    )
