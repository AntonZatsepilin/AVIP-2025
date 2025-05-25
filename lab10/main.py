import numpy as np
import scipy.signal as signal
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Parameters
WINDOW = "hann"
N_FFT = 4096
HOP_LENGTH = 512
FORMANT_PEAKS = 3
F0_MIN = 50  # Hz
F0_MAX = 1000  # Hz
F0_STEP = 10  # Hz
THRESHOLD_RATIO = 0.1  # relative magnitude threshold for harmonics


def load_audio(path):
    # Load as mono
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def compute_spectrogram(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    return S, freqs, times


def save_spectrogram(S, freqs, times, out_png, formants=None):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        sr=None,
        x_coords=times,
        y_coords=freqs,
        x_axis="time",
        y_axis="log",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (log-frequency)")
    if formants is not None:
        for f in formants:
            plt.axhline(
                f, color="red", linestyle="--", linewidth=3, label=f"{f:.1f} Hz"
            )
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def find_freq_range(S, freqs, threshold_db=-40):
    # Convert to dB
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mask = S_db > threshold_db
    if not np.any(mask):
        return 0, 0
    f_present = freqs[np.any(mask, axis=1)]
    return f_present.min(), f_present.max()


def find_timbre_f0(S, freqs):
    # Sum spectrum over time to get average
    spectrum = S.mean(axis=1)
    # Normalize
    spec_norm = spectrum / np.max(spectrum)
    best_f0, best_count = 0, 0
    for f0 in np.arange(F0_MIN, F0_MAX + 1, F0_STEP):
        count = 0
        for k in range(2, int((freqs[-1] // f0)) + 1):
            fk = k * f0
            idx = np.argmin(np.abs(freqs - fk))
            if spec_norm[idx] > THRESHOLD_RATIO:
                count += 1
        if count > best_count:
            best_f0, best_count = f0, count
    return best_f0, best_count


def find_formants(S, freqs, num_formants=FORMANT_PEAKS):
    # Average spectrum
    spectrum = S.mean(axis=1)
    # Find peaks
    peaks, properties = signal.find_peaks(
        spectrum, distance=(40 / (freqs[1] - freqs[0]))
    )
    peak_powers = spectrum[peaks]
    # Sort peaks by power
    idx = np.argsort(peak_powers)[-num_formants:][::-1]
    formant_freqs = freqs[peaks[idx]]
    return formant_freqs


if __name__ == "__main__":
    files = {
        "A": "src/a.wav",
        "I": "src/i.wav",
        "Bark": "src/dog.wav",
    }

    for label, path in files.items():
        print(f"Processing {label} from {path}")
        y, sr = load_audio(path)
        S, freqs, times = compute_spectrogram(y, sr)

        f_min, f_max = find_freq_range(S, freqs)
        print(f"{label}: min freq = {f_min:.1f} Hz, max freq = {f_max:.1f} Hz")

        f0, overtones = find_timbre_f0(S, freqs)
        print(f"{label}: timbral fundamental = {f0} Hz with {overtones} overtones")

        formants = find_formants(S, freqs)
        print(f"{label}: formants = {', '.join(f'{f:.1f}Hz' for f in formants)}")

        save_spectrogram(
            S, freqs, times, f"output/spec_{label}.png", formants=formants
        )
        print("-" * 40)