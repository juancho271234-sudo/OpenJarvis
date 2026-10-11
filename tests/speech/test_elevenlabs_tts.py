"""ElevenLabs TTS backend tests (no network: httpx is patched)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.registry import TTSRegistry

_VOICE = "D09EpJbk4um1HKSpeTSc"


def _response(content: bytes = b"", data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.json.return_value = data or {}
    return resp


def test_elevenlabs_registered():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    TTSRegistry.register_value("elevenlabs", ElevenLabsTTSBackend)
    assert TTSRegistry.contains("elevenlabs")


def test_elevenlabs_health_requires_key(monkeypatch):
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTSBackend().health() is False
    assert ElevenLabsTTSBackend(api_key="k").health() is True


def test_elevenlabs_synthesize_request_shape():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    backend = ElevenLabsTTSBackend(api_key="fake-key")
    with patch(
        "openjarvis.speech.elevenlabs_tts.httpx.post",
        return_value=_response(content=b"mp3-bytes"),
    ) as post:
        result = backend.synthesize("hola", voice_id=_VOICE)

    args, kwargs = post.call_args
    assert args[0] == f"https://api.elevenlabs.io/v1/text-to-speech/{_VOICE}"
    assert kwargs["headers"]["xi-api-key"] == "fake-key"
    assert kwargs["params"] == {"output_format": "mp3_44100_128"}
    assert kwargs["json"] == {"text": "hola", "model_id": "eleven_multilingual_v2"}
    assert result.audio == b"mp3-bytes"
    assert result.format == "mp3"
    assert result.voice_id == _VOICE


def test_elevenlabs_speed_sent_and_clamped():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    backend = ElevenLabsTTSBackend(api_key="k")
    with patch(
        "openjarvis.speech.elevenlabs_tts.httpx.post",
        return_value=_response(content=b"x"),
    ) as post:
        backend.synthesize("hi", voice_id=_VOICE, speed=2.0)
    assert post.call_args.kwargs["json"]["voice_settings"] == {"speed": 1.2}


def test_elevenlabs_requires_key_and_voice(monkeypatch):
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        ElevenLabsTTSBackend().synthesize("hi", voice_id=_VOICE)
    with pytest.raises(RuntimeError, match="voice ID"):
        ElevenLabsTTSBackend(api_key="k").synthesize("hi")


def test_elevenlabs_available_voices():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    backend = ElevenLabsTTSBackend(api_key="k")
    with patch(
        "openjarvis.speech.elevenlabs_tts.httpx.get",
        return_value=_response(data={"voices": [{"voice_id": "a"}, {"voice_id": "b"}]}),
    ):
        assert backend.available_voices() == ["a", "b"]
