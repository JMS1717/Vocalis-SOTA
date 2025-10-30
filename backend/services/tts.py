"""Text-to-Speech Service utilities."""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, Generator, Optional

import requests

logger = logging.getLogger(__name__)

class TTSClient:
    """Client for communicating with configurable TTS providers."""

    def __init__(
        self,
        api_endpoint: str = "http://localhost:5005/v1/audio/speech",
        model: str = "tts-1",
        voice: Optional[str] = None,
        output_format: str = "wav",
        speed: float = 1.0,
        timeout: int = 60,
        chunk_size: int = 4096,
        provider: str = "openai-compatible",
        api_key: Optional[str] = None,
        inference_params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the TTS client."""

        self.api_endpoint = api_endpoint
        self.model = model
        self.voice = voice
        self.output_format = output_format
        self.speed = speed
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.provider = provider
        self.api_key = api_key
        self.inference_params = inference_params or {}
        self.extra_headers = extra_headers or {}

        # Persistent HTTP session for keep-alive reuse.
        self.session = requests.Session()
        self.session.headers.update(self._build_default_headers())

        # State tracking
        self.is_processing = False
        self.last_processing_time = 0

        logger.info(
            "Initialized TTS client endpoint=%s model=%s provider=%s",
            api_endpoint,
            model,
            provider,
        )

    def _build_default_headers(self) -> Dict[str, str]:
        """Build the default headers for HTTP requests."""

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        if self.provider.startswith("huggingface"):
            # Hugging Face TTS endpoints commonly return binary audio.
            headers.setdefault("Accept", "application/octet-stream")

        if self.extra_headers:
            headers.update(self.extra_headers)

        return headers

    def _build_payload(self, text: str) -> Dict[str, Any]:
        """Construct a provider-specific request payload."""

        if self.provider.startswith("huggingface"):
            parameters: Dict[str, Any] = dict(self.inference_params)

            if self.voice:
                parameters.setdefault("voice", self.voice)

            if self.output_format:
                parameters.setdefault("format", self.output_format)

            if self.speed and self.speed != 1.0:
                parameters.setdefault("speed", self.speed)

            return {
                "inputs": text,
                "parameters": {k: v for k, v in parameters.items() if v is not None},
            }

        payload = {
            "model": self.model,
            "input": text,
            "response_format": self.output_format,
        }

        if self.voice is not None:
            payload["voice"] = self.voice

        if self.speed is not None:
            payload["speed"] = self.speed

        return payload

    def _decode_audio_payload(self, payload: Any) -> Optional[bytes]:
        """Extract audio bytes from a JSON payload."""

        if payload is None:
            return None

        if isinstance(payload, dict):
            candidates = [
                payload.get("audio"),
                payload.get("audio_base64"),
                payload.get("generated_audio"),
                payload.get("b64_audio"),
                payload.get("data"),
            ]

            for candidate in candidates:
                audio_bytes = self._decode_audio_payload(candidate)
                if audio_bytes:
                    return audio_bytes

        if isinstance(payload, (list, tuple)):
            for item in payload:
                audio_bytes = self._decode_audio_payload(item)
                if audio_bytes:
                    return audio_bytes

        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)

        if isinstance(payload, str):
            try:
                return base64.b64decode(payload)
            except (ValueError, TypeError):
                return None

        return None

    def _extract_audio_response(self, response: requests.Response) -> bytes:
        """Return raw audio bytes from a HTTP response."""

        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            json_payload = response.json()
            audio_bytes = self._decode_audio_payload(json_payload)

            if audio_bytes is None:
                raise ValueError("TTS response did not include audio data")

            return audio_bytes

        return response.content

    def _iter_stream_chunks(self, response: requests.Response) -> Generator[bytes, None, None]:
        """Yield audio chunks from a streaming HTTP response."""

        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            for line in response.iter_lines(decode_unicode=False):
                if not line:
                    continue

                if line.startswith(b"data:"):
                    data = line[len(b"data:") :].strip()

                    if data in (b"[DONE]", b""):
                        continue

                    try:
                        payload = json.loads(data.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        logger.debug("Skipping non-JSON SSE payload: %s", data)
                        continue

                    audio_bytes = self._decode_audio_payload(payload)

                    if audio_bytes:
                        yield audio_bytes

            return

        transfer_encoding = response.headers.get("transfer-encoding", "").lower()

        if "chunked" in transfer_encoding:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    yield chunk
            return

        # Fallback: non-streaming response, split for compatibility.
        audio_data = self._extract_audio_response(response)

        total_chunks = (len(audio_data) + self.chunk_size - 1) // self.chunk_size

        for index in range(total_chunks):
            start_idx = index * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(audio_data))
            yield audio_data[start_idx:end_idx]
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        self.is_processing = True
        start_time = time.time()
        
        try:
            payload = self._build_payload(text)

            logger.info(
                "Sending TTS request chars=%d provider=%s endpoint=%s",
                len(text),
                self.provider,
                self.api_endpoint,
            )

            response = self.session.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            audio_data = self._extract_audio_response(response)

            # Calculate processing time
            self.last_processing_time = time.time() - start_time
            
            logger.info(
                "Received TTS response after %.2fs size=%d bytes",
                self.last_processing_time,
                len(audio_data),
            )
            
            return audio_data
            
        except requests.RequestException as e:
            logger.error("TTS API request error: %s", e)
            raise
        except Exception as e:
            logger.error("TTS processing error: %s", e, exc_info=True)
            raise
        finally:
            self.is_processing = False
    
    def stream_text_to_speech(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream audio data from the TTS API.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Chunks of audio data
        """
        self.is_processing = True
        start_time = time.time()
        
        try:
            payload = self._build_payload(text)

            logger.info(
                "Sending streaming TTS request chars=%d provider=%s endpoint=%s",
                len(text),
                self.provider,
                self.api_endpoint,
            )

            with self.session.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()

                for chunk in self._iter_stream_chunks(response):
                    if chunk:
                        yield chunk

            # Calculate processing time
            self.last_processing_time = time.time() - start_time
            logger.info(
                "Completed TTS streaming after %.2fs",
                self.last_processing_time,
            )
            
        except requests.RequestException as e:
            logger.error("TTS API streaming request error: %s", e)
            raise
        except Exception as e:
            logger.error("TTS streaming error: %s", e, exc_info=True)
            raise
        finally:
            self.is_processing = False
    
    async def async_text_to_speech(self, text: str) -> bytes:
        """
        Asynchronously generate audio data from the TTS API.
        
        This method provides asynchronous TTS capability by running
        the synchronous method in a thread.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Complete audio data as bytes
        """
        self.is_processing = True
        
        try:
            # Get complete audio data
            audio_data = await asyncio.to_thread(self.text_to_speech, text)
            return audio_data
        except Exception as e:
            logger.error("Async TTS error: %s", e)
            raise
        finally:
            self.is_processing = False

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict containing the current configuration
        """
        return {
            "api_endpoint": self.api_endpoint,
            "model": self.model,
            "voice": self.voice,
            "output_format": self.output_format,
            "speed": self.speed,
            "timeout": self.timeout,
            "chunk_size": self.chunk_size,
            "provider": self.provider,
            "inference_params": dict(self.inference_params),
            "extra_headers": dict(self.extra_headers),
            "is_processing": self.is_processing,
            "last_processing_time": self.last_processing_time,
        }
