"""Speech-to-Text transcription service for local Kyutai models."""

from __future__ import annotations

import io
import logging
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """Run automatic speech recognition with local Hugging Face models."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        sample_rate: int = 16_000,
        generation_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.generation_config = generation_config or {"max_new_tokens": 256}
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

        self.model_source = self._resolve_model_source(self.model_id)

        logger.info(
            "Loading speech model %s from %s on %s (dtype=%s)",
            self.model_id,
            self.model_source,
            self.device,
            self.torch_dtype,
        )

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}

        if self.cache_dir:
            load_kwargs["cache_dir"] = str(self.cache_dir)
            processor_kwargs["cache_dir"] = str(self.cache_dir)

        self.processor = AutoProcessor.from_pretrained(
            self.model_source,
            **processor_kwargs,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_source,
            **load_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

        self.is_processing = False

    def _resolve_model_source(self, model_id: str) -> str:
        if not self.cache_dir:
            return model_id

        safe_name = model_id.replace("/", "__")
        candidate = self.cache_dir / safe_name

        if candidate.exists():
            return str(candidate)

        logger.info(
            "Local cache for %s not found at %s; falling back to Hugging Face hub",
            model_id,
            candidate,
        )
        return model_id

    def _resolve_dtype(self, torch_dtype: Optional[str]) -> torch.dtype:
        if torch_dtype:
            try:
                return getattr(torch, torch_dtype)
            except AttributeError as exc:  # pragma: no cover - defensive path
                raise ValueError(f"Unsupported torch dtype: {torch_dtype}") from exc

        if self.device.startswith("cuda"):
            return torch.float16
        return torch.float32

    @staticmethod
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.float32:
            normalized = audio
        elif audio.dtype == np.int16:
            normalized = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            normalized = audio.astype(np.float32) / 2147483648.0
        else:
            normalized = audio.astype(np.float32)
            max_val = np.max(np.abs(normalized))
            if max_val > 0:
                normalized /= max_val

        if normalized.ndim > 1:
            normalized = normalized.mean(axis=1)

        return normalized.astype(np.float32, copy=False)

    def _decode_wav_bytes(self, audio_bytes: Union[bytes, bytearray]) -> Tuple[np.ndarray, int]:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                framerate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
        except wave.Error as exc:  # pragma: no cover - invalid input
            raise ValueError("Unsupported or corrupted WAV data") from exc

        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8)
            data = data.astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:  # pragma: no cover - defensive coding for unusual widths
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        return data, framerate

    def _resample_audio(self, audio: np.ndarray, original_rate: int) -> np.ndarray:
        if original_rate == self.sample_rate or audio.size == 0:
            return audio.astype(np.float32)

        duration = audio.shape[0] / float(original_rate)
        target_length = max(int(round(duration * self.sample_rate)), 1)

        original_times = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False, dtype=np.float32)
        target_times = np.linspace(0.0, duration, num=target_length, endpoint=False, dtype=np.float32)

        resampled = np.interp(target_times, original_times, audio.astype(np.float32))
        return resampled.astype(np.float32)

    def _prepare_audio(self, audio: Union[np.ndarray, bytes, bytearray]) -> np.ndarray:
        if isinstance(audio, (bytes, bytearray)):
            normalized, original_rate = self._decode_wav_bytes(audio)
        elif isinstance(audio, np.ndarray):
            if audio.dtype == np.uint8:
                normalized, original_rate = self._decode_wav_bytes(audio.tobytes())  # pragma: no cover - legacy path
            else:
                normalized = self._normalize_audio(audio)
                original_rate = self.sample_rate
        else:  # pragma: no cover - defensive
            raise TypeError("Unsupported audio input type")

        if original_rate != self.sample_rate:
            normalized = self._resample_audio(normalized, original_rate)

        return normalized

    def transcribe(self, audio: Union[np.ndarray, bytes, bytearray]) -> tuple[str, Dict[str, Any]]:
        start_time = time.time()
        self.is_processing = True

        try:
            float_audio = np.ascontiguousarray(self._prepare_audio(audio), dtype=np.float32)

            inputs = self.processor(
                audio=float_audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )

            model_inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }

            if "input_features" in model_inputs:
                input_features = model_inputs.pop("input_features")
            elif "input_values" in model_inputs:
                input_features = model_inputs.pop("input_values")
            else:
                raise ValueError("Processor did not return input features")

            generate_kwargs = dict(self.generation_config)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_features,
                    **generate_kwargs,
                )

            transcript = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            processing_time = time.time() - start_time

            metadata = {
                "processing_time": processing_time,
                "model_id": self.model_id,
                "device": self.device,
                "dtype": str(self.torch_dtype).split(".")[-1],
                "num_tokens": int(generated_ids.shape[-1]) if generated_ids.ndim > 1 else 0,
            }

            return transcript.strip(), metadata

        except Exception as exc:
            logger.error("Transcription error: %s", exc, exc_info=True)
            return "", {"error": str(exc)}

        finally:
            self.is_processing = False

    def close(self) -> None:
        """Release model resources."""

        try:
            model = getattr(self, "model", None)
            if model is not None:
                model.to("cpu")
            if torch.cuda.is_available():  # pragma: no cover - requires GPU
                torch.cuda.empty_cache()
        finally:
            self.model = None  # type: ignore[assignment]
            self.processor = None  # type: ignore[assignment]
            self.is_processing = False

    def transcribe_streaming(self, audio_generator: Any):
        raise NotImplementedError("Streaming transcription is not yet implemented for Kyutai models")

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_source,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype).split(".")[-1],
            "sample_rate": self.sample_rate,
            "generation_config": self.generation_config,
            "is_processing": self.is_processing,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
