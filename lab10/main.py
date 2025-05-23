# -*- coding: utf-8 -*-
"""
main_with_plots.py — Лабораторная работа №10: Обработка голоса с дополнительными графиками
"""
import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

SRC_DIR     = 'src'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_spectrogram(y, sr, title, outpath):
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_f0_contour(y, sr, title, outpath):
    f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr,
                     frame_length=2048, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    plt.figure(figsize=(8,4))
    plt.plot(times, f0, label='F0 contour')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_spectral_peaks(y, sr, harmonics, formants, title, outpath):
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec_avg = D.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_db = librosa.amplitude_to_db(spec_avg, ref=np.max)
    plt.figure(figsize=(8,4))
    plt.plot(freqs, spec_db, label='Average spectrum (dB)')

    for k, h in enumerate(harmonics, start=1):
        plt.axvline(x=h, linestyle='--', label=f'Harmonic {k}: {h:.1f} Hz')

    for i, f in enumerate(formants, start=1):
        plt.axvline(x=f, color='red', linestyle=':', label=f'Formant {i}: {f:.1f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def min_max_frequency(y, sr, threshold_db=-60):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mean_spec = S_db.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = mean_spec > threshold_db
    if not mask.any():
        return 0.0, 0.0
    fmin = freqs[mask].min()
    fmax = freqs[mask].max()
    return fmin, fmax


def estimate_f0_and_overtones(y, sr, fmin=50, fmax=800):
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=2048, hop_length=512)
    f0_med = np.nanmedian(f0)
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_avg = D.mean(axis=1)

    harmonics = []
    k = 1
    while True:
        h_freq = f0_med * k
        if h_freq >= sr/2:
            break
        idx = np.argmin(np.abs(freqs - h_freq))
        if spec_avg[idx] > 0.5 * spec_avg.max():
            harmonics.append(h_freq)
        k += 1
    return f0_med, harmonics


def estimate_formants(y, sr, n_formants=3, lpc_order=16):
    a = librosa.lpc(y, order=lpc_order)
    roots = np.roots(a)
    angles = np.angle(roots)
    freqs = angles * sr / (2 * np.pi)
    freqs = freqs[freqs > 0]
    freqs = np.sort(freqs)
    return freqs[:n_formants]


def main():
    files = glob.glob(os.path.join(SRC_DIR, '*.wav'))
    report = []

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        y, sr = librosa.load(path, sr=None, mono=True)
        print(f"Processing {name} (sr={sr}, length={len(y)/sr:.2f}s)")

        plot_spectrogram(y, sr, f'Spectrogram: {name}', \
                         os.path.join(RESULTS_DIR, f'spec_{name}.png'))

        plot_f0_contour(y, sr, f'F0 Contour: {name}', \
                         os.path.join(RESULTS_DIR, f'f0_{name}.png'))

        fmin, fmax = min_max_frequency(y, sr)

        f0_med, harmonics = estimate_f0_and_overtones(y, sr)

        formants = estimate_formants(y, sr)

        plot_spectral_peaks(y, sr, harmonics, formants, \
                            f'Peaks: {name}', \
                            os.path.join(RESULTS_DIR, f'peaks_{name}.png'))

        report.append({
            'name': name,
            'fmin': fmin,
            'fmax': fmax,
            'f0': f0_med,
            'overtones': harmonics,
            'formants': formants.tolist()
        })

    with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        for item in report:
            f.write(f"File: {item['name']}\n")
            f.write(f"Min freq: {item['fmin']:.1f} Hz, Max freq: {item['fmax']:.1f} Hz\n")
            f.write(f"Fundamental (median): {item['f0']:.1f} Hz\n")
            f.write(f"Overtones: {', '.join(f'{h:.1f}' for h in item['overtones'])}\n")
            f.write(f"Formants: {', '.join(f'{f:.1f}' for f in item['formants'])} Hz\n")
            f.write("\n")
    print(f"Report saved to {os.path.join(RESULTS_DIR, 'report.txt')}")

if __name__ == '__main__':
    main()