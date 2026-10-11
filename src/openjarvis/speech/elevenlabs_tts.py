"""ElevenLabs text-to-speech backend.

Uses the ElevenLabs REST API for high-quality voice synthesis, including
custom, cloned and Voice Library voices.
Requires the ELEVENLABS_API_KEY environment variable (or ``api_key`` argument).

Voices from the public Voice Library must be added to "My Voices" in your
ElevenLabs account before the API can use them.
"""

from __future__ import annotations

import os
from typing import List

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
_DEFAULT_MODEL = "eleven_multilingual_v2"
_MP3_FORMAT = "mp3_44100_128"
# ElevenLabs accepts a speaking-rate multiplier in this range.
_MIN_SPEED = 0.7
_MAX_SPEED = 1.2


def _elevenlabs_synthesize(
    api_key: str,
    text: str,
    voice_id: str,
    model: str = _DEFAULT_MODEL,
    speed: float = 1.0,
) -> bytes:
    """Call the ElevenLabs TTS API and return MP3 audio bytes."""
    payload: dict = {"text": text, "model_id": model}
    if speed != 1.0:
        payload["voice_settings"] = {
            "speed": max(_MIN_SPEED, min(_MAX_SPEED, speed)),
        }
    resp = httpx.post(
        f"{_ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id}",
        params={"output_format": _MP3_FORMAT},
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.content


@TTSRegistry.register("elevenlabs")
class ElevenLabsTTSBackend(TTSBackend):
    """ElevenLabs TTS backend — natural, multilingual, custom voices."""

    backend_id = "elevenlabs"

    def __init__(self, *, api_key: str = "", model: str = "") -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._model = (
            model or os.environ.get("ELEVENLABS_MODEL", "") or _DEFAULT_MODEL
        )

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        if not voice_id:
            raise RuntimeError(
                "ElevenLabs needs a voice ID: set speech.voice_id in config.toml"
            )

        audio = _elevenlabs_synthesize(
            self._api_key,
            text,
            voice_id=voice_id,
            model=self._model,
            speed=speed,
        )

        # The API is always asked for MP3, whatever ``output_format`` was
        # requested, so report what the bytes actually are.
        return TTSResult(
            audio=audio,
            format="mp3",
            sample_rate=44100,
            voice_id=voice_id,
            metadata={"backend": "elevenlabs", "model": self._model},
        )

    def available_voices(self) -> List[str]:
        if not self._api_key:
            return []
        resp = httpx.get(
            f"{_ELEVENLABS_API_BASE}/v1/voices",
            headers={"xi-api-key": self._api_key},
            timeout=30.0,
        )
        resp.raise_for_status()
        return [v["voice_id"] for v in resp.json().get("voices", [])]

    def health(self) -> bool:
        return bool(self._api_key)
