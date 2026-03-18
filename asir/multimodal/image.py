import io
import base64
from typing import Optional
import numpy as np
import dspy


def generate_spectrogram_image(signal, title: str = "Spectrogram") -> Optional[dspy.Image]:
    """
    [Phase 2] 從 RawSignal 生成 Mel-like 頻譜圖，回傳 dspy.Image。
    讓所有 vision LM（GPT-4o、Claude、Gemini）都能「看到」音訊結構。

    ★ 不依賴 librosa — 只用 numpy + matplotlib。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非互動 backend（伺服器/CI 友善）
        import matplotlib.pyplot as plt

        samples = np.array(signal.samples[0])
        sr = signal.sample_rate

        # STFT
        frame_size = min(256, len(samples))
        hop = frame_size // 2
        window = np.hanning(frame_size)
        frames = []
        for i in range(0, len(samples) - frame_size, hop):
            frame = samples[i:i + frame_size]
            windowed = frame * window
            spectrum = np.abs(np.fft.rfft(windowed))
            frames.append(spectrum)

        if not frames:
            return None

        spectrogram = np.array(frames).T
        spectrogram_db = 10 * np.log10(spectrogram + 1e-10)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=100)
        ax.imshow(spectrogram_db, aspect='auto', origin='lower',
                  cmap='magma',
                  extent=[0, signal.duration_ms, 0, sr / 2])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        plt.colorbar(ax.images[0], ax=ax, label='dB')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        img_b64 = base64.b64encode(buf.read()).decode()
        return dspy.Image(f"data:image/png;base64,{img_b64}")
    except ImportError:
        # matplotlib 未安裝 → 優雅降級
        return None
    except Exception:
        return None


def generate_mfcc_plot(signal, n_coeffs: int = 13) -> Optional[dspy.Image]:
    """
    [Phase 2] 生成簡化的 MFCC 視覺化圖（不依賴 librosa）。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        samples = np.array(signal.samples[0])
        frame_size = min(256, len(samples))
        hop = frame_size // 2
        window = np.hanning(frame_size)

        # 計算簡化版 MFCC：對頻譜取 DCT 的近似
        mfcc_frames = []
        for i in range(0, len(samples) - frame_size, hop):
            frame = samples[i:i + frame_size]
            windowed = frame * window
            power_spectrum = np.abs(np.fft.rfft(windowed)) ** 2
            # 簡化的 Mel 濾波 + log + DCT（用 FFT 近似 DCT）
            log_spectrum = np.log(power_spectrum + 1e-10)
            mfcc = np.fft.dct_type2 if hasattr(np.fft, 'dct_type2') else None
            # 用 real FFT 近似 DCT
            ceps = np.fft.irfft(log_spectrum)[:n_coeffs]
            mfcc_frames.append(ceps)

        if not mfcc_frames:
            return None

        mfcc_matrix = np.array(mfcc_frames).T

        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=100)
        ax.imshow(mfcc_matrix, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Frame')
        ax.set_ylabel('MFCC Coefficient')
        ax.set_title('MFCC Features')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        img_b64 = base64.b64encode(buf.read()).decode()
        return dspy.Image(f"data:image/png;base64,{img_b64}")
    except Exception:
        return None
