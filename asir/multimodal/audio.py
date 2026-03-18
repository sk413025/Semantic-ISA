import io
import base64
from typing import Optional
import numpy as np
import dspy


def raw_signal_to_audio(signal) -> Optional[dspy.Audio]:
    """
    [Phase 1] 將 RawSignal 的 numpy 波形轉換為 dspy.Audio。
    讓支援音訊的多模態 LM（Gemini、GPT-4o-audio）能「直接聽」原始音訊。

    ★ 若 dspy.Audio.from_array 不可用，回退為手動 WAV 編碼。
    """
    try:
        import struct as _struct
        samples = np.array(signal.samples[0], dtype=np.float32)
        # 歸一化到 [-1, 1]
        peak = np.max(np.abs(samples)) + 1e-10
        samples_norm = samples / peak

        # 嘗試 dspy.Audio.from_array（需要 soundfile 套件）
        try:
            if hasattr(dspy.Audio, 'from_array'):
                return dspy.Audio.from_array(samples_norm, signal.sample_rate)
        except Exception:
            pass  # soundfile 未安裝，回退到手動編碼

        # 回退：手動編碼為 WAV base64（不需要額外套件）
        pcm16 = (samples_norm * 32767).astype(np.int16)
        buf = io.BytesIO()
        n = len(pcm16)
        buf.write(b'RIFF')
        buf.write(_struct.pack('<I', 36 + n * 2))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(_struct.pack('<IHHIIHH', 16, 1, 1, signal.sample_rate,
                               signal.sample_rate * 2, 2, 16))
        buf.write(b'data')
        buf.write(_struct.pack('<I', n * 2))
        buf.write(pcm16.tobytes())
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        return dspy.Audio(data=encoded, audio_format="wav")
    except Exception:
        return None
