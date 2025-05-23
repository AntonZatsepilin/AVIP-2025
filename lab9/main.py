import os
import subprocess
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import wiener, butter, filtfilt
from scipy.ndimage import maximum_filter
from scipy.io.wavfile import write

SRC_DIR     = "src"
RESULTS_DIR = "results"
os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

REC_PATH    = os.path.join(SRC_DIR, "guitar.wav")
FILTERED_WAV= os.path.join(RESULTS_DIR, "denoised.wav")

def record_audio(duration=5, device=None):
    """
    Запись из микрофона в WAV через ffmpeg:
    duration — время в секундах;
    device — имя устройства (None → default ALSA).
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "alsa" if device is None else "dshow",
        "-i", device or "default",
        "-t", str(duration),
        "-ac", "1",         # один канал
        REC_PATH
    ]
    subprocess.run(cmd, check=True)
    print(f"[✓] Запись сохранена: {REC_PATH}")

def plot_spectrogram(y, sr, title, out_path):
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] Спектрограмма сохранена: {out_path}")

def denoise_signal(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает два варианта очищенного сигнала:
    y_wien — Wiener-фильтр;
    y_lp   — низкочастотный фильтр Butterworth.
    """

    y_wien = wiener(y, mysize=29)

    b, a = butter(4, 0.1, btype='low')  # 0.1*Nyquist
    y_lp = filtfilt(b, a, y)
    return y_wien, y_lp

def find_time_freq_peaks(y, sr, n_fft=1024, hop_length=256, dt=0.1, df=50):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    t_win = int(dt * sr / hop_length)
    f_res = freqs[1] - freqs[0]
    f_win = max(int(df / f_res), 1)

    local_max = maximum_filter(S, size=(f_win, t_win))
    peaks = np.argwhere(S == local_max)

    peaks_list = [(times[t], freqs[f], S[f, t]) for f, t in peaks]
    peaks_list.sort(key=lambda x: x[2], reverse=True)
    return peaks_list[:10]

def main():

    y, sr = librosa.load(REC_PATH, sr=None, mono=True)
    print(f"[i] Загружено: {REC_PATH}, sr={sr}, длительность={len(y)/sr:.2f}s")

    plot_spectrogram(y, sr, 'Original', os.path.join(RESULTS_DIR, 'spec_before.png'))

    y_wien, y_lp = denoise_signal(y)

    write(FILTERED_WAV.replace('.wav', '_wiener.wav'), sr, (y_wien * 32767).astype(np.int16))
    print(f"[✓] Wiener сохранён: {FILTERED_WAV.replace('.wav', '_wiener.wav')}")
    write(FILTERED_WAV.replace('.wav', '_lowpass.wav'), sr, (y_lp * 32767).astype(np.int16))
    print(f"[✓] Lowpass сохранён: {FILTERED_WAV.replace('.wav', '_lowpass.wav')}")

    plot_spectrogram(y_wien, sr, 'Wiener denoised', os.path.join(RESULTS_DIR, 'spec_wiener.png'))
    plot_spectrogram(y_lp, sr,   'Lowpass denoised', os.path.join(RESULTS_DIR, 'spec_lowpass.png'))

    peaks = find_time_freq_peaks(y_wien, sr)
    with open(os.path.join(RESULTS_DIR, 'peaks.txt'), 'w') as f:
        for t, fr, e in peaks:
            f.write(f"{t:.3f}s @ {fr:.1f}Hz — energy={e:.2f}\n")
    print("[✓] Пики энергии сохранены: peaks.txt")

if __name__ == '__main__':
    main()