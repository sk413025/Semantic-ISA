import base64
import io
from typing import Optional

import dspy
import numpy as np


def raw_signal_to_audio(signal) -> Optional[dspy.Audio]:
    """
    [Phase 1] Convert a RawSignal waveform into dspy.Audio so audio-capable
    multimodal models can listen directly to the original signal.

    If dspy.Audio.from_array is unavailable, fall back to manual WAV encoding.
    """

    try:
        import struct as _struct

        samples = np.array(signal.samples[0], dtype=np.float32)

        # Normalize to [-1, 1].
        peak = np.max(np.abs(samples)) + 1e-10
        samples_norm = samples / peak

        # Try dspy.Audio.from_array (requires the soundfile package).
        try:
            if hasattr(dspy.Audio, "from_array"):
                return dspy.Audio.from_array(samples_norm, signal.sample_rate)
        except Exception:
            pass  # soundfile is unavailable; fall back to manual encoding.

        # Fallback: manually encode the waveform as base64 WAV without extra dependencies.
        pcm16 = (samples_norm * 32767).astype(np.int16)
        buf = io.BytesIO()
        n = len(pcm16)
        buf.write(b"RIFF")
        buf.write(_struct.pack("<I", 36 + n * 2))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(
            _struct.pack(
                "<IHHIIHH",
                16,
                1,
                1,
                signal.sample_rate,
                signal.sample_rate * 2,
                2,
                16,
            )
        )
        buf.write(b"data")
        buf.write(_struct.pack("<I", n * 2))
        buf.write(pcm16.tobytes())
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        return dspy.Audio(data=encoded, audio_format="wav")
    except Exception:
        return None
