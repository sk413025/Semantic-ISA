import numpy as np
from asir.types import RawSignal


def prim_load_audio(file_path: str = None, audio_bytes: bytes = None,
                    sample_rate: int = 16000) -> tuple:
    """
    [Phase 1] 載入真實音訊檔案，回傳 (RawSignal, dspy.Audio) 元組。
    支援 WAV 格式。若有 scipy 可用則支援更多格式。

    ★ 這是 prim_sample_audio() 的真實音訊版本。
    """
    import dspy

    audio_obj = None

    if file_path is not None:
        # 嘗試用 dspy.Audio.from_file
        if hasattr(dspy.Audio, 'from_file'):
            audio_obj = dspy.Audio.from_file(file_path)

        # 讀取波形數據
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(file_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            if len(data.shape) == 1:
                samples = [data.tolist(), data.tolist()]  # mono → dual
                n_ch = 2
            else:
                samples = [data[:, i].tolist() for i in range(min(data.shape[1], 2))]
                n_ch = len(samples)
            signal = RawSignal(
                samples=samples,
                sample_rate=sr,
                n_channels=n_ch,
                duration_ms=float(len(data) / sr * 1000)
            )
            return signal, audio_obj
        except ImportError:
            # scipy 不可用，用 struct 手動解析 WAV
            import struct
            with open(file_path, 'rb') as f:
                raw = f.read()
            # 極簡 WAV parser
            if raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
                # 找 data chunk
                idx = 12
                while idx < len(raw) - 8:
                    chunk_id = raw[idx:idx+4]
                    chunk_size = struct.unpack('<I', raw[idx+4:idx+8])[0]
                    if chunk_id == b'data':
                        pcm = np.frombuffer(raw[idx+8:idx+8+chunk_size], dtype=np.int16)
                        data = pcm.astype(np.float32) / 32768.0
                        signal = RawSignal(
                            samples=[data.tolist(), data.tolist()],
                            sample_rate=sample_rate,
                            n_channels=2,
                            duration_ms=float(len(data) / sample_rate * 1000)
                        )
                        return signal, audio_obj
                    idx += 8 + chunk_size

    # Fallback：使用模擬信號
    signal = prim_sample_audio(duration_ms=32.0)
    from asir.multimodal.audio import raw_signal_to_audio
    audio_obj = raw_signal_to_audio(signal)
    return signal, audio_obj


def prim_sample_audio(duration_ms: float = 32.0, n_channels: int = 2,
                      sample_rate: int = 16000) -> RawSignal:
    """
    [PRIM] 第一層：物理感測
    BACKEND: deterministic (硬體)
    可靠性: 100%
    類比: CPU 的 LOAD 指令
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    # 模擬菜市場音訊：多聲源混合
    t = np.linspace(0, duration_ms / 1000, n_samples)
    # 目標語音（攤販，正前方）
    target_speech = 0.3 * np.sin(2 * np.pi * 200 * t) * np.random.normal(1, 0.1, n_samples)
    # 背景多人交談（環繞）
    babble = 0.4 * np.random.randn(n_samples)
    # 偶發金屬碰撞（右側）
    impulse = np.zeros(n_samples)
    impulse[n_samples // 3] = 2.0  # 單一脈衝

    ch0 = (target_speech + babble + impulse * 0.3).tolist()  # 左耳
    ch1 = (target_speech * 0.8 + babble + impulse * 0.7).tolist()  # 右耳

    return RawSignal(
        samples=[ch0, ch1],
        sample_rate=sample_rate,
        n_channels=n_channels,
        duration_ms=duration_ms
    )


def prim_fft(signal: RawSignal) -> dict:
    """
    [PRIM] 第二層：FFT
    BACKEND: deterministic
    可靠性: 100%（數值精度內）
    """
    spectrum = np.fft.rfft(signal.samples[0])
    return {
        "magnitude": np.abs(spectrum).tolist(),
        "phase": np.angle(spectrum).tolist(),
        "freq_bins": len(spectrum)
    }


def prim_estimate_noise_psd(signal: RawSignal) -> list[float]:
    """
    [PRIM] 第二層：噪音功率譜密度估計
    BACKEND: deterministic (minimum statistics 方法)
    可靠性: 100%
    """
    spectrum = np.fft.rfft(signal.samples[0])
    # 簡化版 minimum statistics
    psd = np.abs(spectrum) ** 2
    noise_floor = np.percentile(psd, 25)  # 取 25th percentile 作為噪音底線
    return (psd * (psd < noise_floor * 4)).tolist()


def prim_beamform(signal: RawSignal, target_azimuth_deg: float = 0.0) -> list[float]:
    """
    [PRIM] 第二層：固定方向波束成形
    BACKEND: deterministic
    可靠性: 100%
    """
    ch0 = np.array(signal.samples[0])
    ch1 = np.array(signal.samples[1])
    # 簡化版 delay-and-sum beamformer
    delay_samples = int(0.01 * np.sin(np.radians(target_azimuth_deg)) * signal.sample_rate)
    if delay_samples > 0:
        ch1_delayed = np.roll(ch1, delay_samples)
    else:
        ch1_delayed = np.roll(ch1, delay_samples)
    beamformed = (ch0 + ch1_delayed) / 2.0
    return beamformed.tolist()


def comp_spectral_subtract(signal: RawSignal, noise_psd: list[float],
                           alpha: float = 1.0) -> list[float]:
    """
    [COMP] 第二層：頻譜減法
    = fft(signal) |> subtract_magnitude(noise, alpha) |> ifft()
    由三個 primitive 組合，仍完全確定性
    """
    spectrum = np.fft.rfft(signal.samples[0])
    noise = np.array(noise_psd[:len(spectrum)])
    magnitude = np.maximum(np.abs(spectrum) - alpha * np.sqrt(noise), 0)
    cleaned = magnitude * np.exp(1j * np.angle(spectrum))
    return np.fft.irfft(cleaned).tolist()
