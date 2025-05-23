import os
import glob
import colorsys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

SRC_DIR = "pictures_src"
DST_DIR = "pictures_results"
os.makedirs(DST_DIR, exist_ok=True)

D = 1
G = 16

GAMMA = 0.5  # Параметр для степенного преобразования

def power_transform(L: np.ndarray, gamma=GAMMA):
    """Применяет степенное преобразование к каналу L."""
    L = np.clip(L, 0.0, 1.0)
    return np.power(L, gamma)
def rgb_to_hls_arr(img: np.ndarray):
    """img: H×W×3 uint8 → возвращает (H,W) массивы H,L,S в [0..1]."""
    hls = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):

        for j in range(img.shape[1]):
            r, g, b = img[i,j] / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            h = max(0.0, min(1.0, h))
            l = max(0.0, min(1.0, l))
            s = max(0.0, min(1.0, s))
            hls[i,j] = (h, l, s)
    return hls[:,:,0], hls[:,:,1], hls[:,:,2]

def hls_to_rgb_arr(H, L, S):
    """H,L,S в [0..1] → возвращает H×W×3 uint8."""
    out = np.zeros((H.shape[0], H.shape[1], 3), dtype=np.uint8)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            h = max(0.0, min(1.0, H[i,j]))
            l = max(0.0, min(1.0, L[i,j]))
            s = max(0.0, min(1.0, S[i,j]))
            r, g, b = colorsys.hls_to_rgb(H[i,j], L[i,j], S[i,j])
            out[i,j] = (
                int(np.clip(r*255, 0, 255)),
                int(np.clip(g*255, 0, 255)),
                int(np.clip(b*255, 0, 255))
            )
    return out

def equalize_histogram(L: np.ndarray):
    """L float [0..1] → L_eq float [0..1] методом выравнивания гистограммы."""
    flat = (L*255).astype(int).ravel()
    hist = np.bincount(flat, minlength=256)
    cdf = hist.cumsum()
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    L_eq = cdf_norm[flat].reshape(L.shape)
    return L_eq

def compute_lbp(image: np.ndarray):
    """Вычисляет карту LBP для полутонового изображения."""
    h, w = image.shape
    if h < 3 or w < 3:
        raise ValueError("Изображение слишком маленькое для LBP (мин. 3x3 пикселя)")
    
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image[i,j]
            code = 0
            code |= (image[i-1,j-1] >= center) << 7
            code |= (image[i-1,j] >= center) << 6
            code |= (image[i-1,j+1] >= center) << 5
            code |= (image[i,j+1] >= center) << 4
            code |= (image[i+1,j+1] >= center) << 3
            code |= (image[i+1,j] >= center) << 2
            code |= (image[i+1,j-1] >= center) << 1
            code |= (image[i,j-1] >= center) << 0
            lbp[i-1,j-1] = code
    return lbp

def compute_lbp_histogram(lbp: np.ndarray):
    """Вычисляет гистограмму LBP."""
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    return hist

def compute_features(hist: np.ndarray):
    """Рассчитывает энтропию и среднее значение гистограммы."""
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    mean = np.mean(hist)
    return entropy, mean


def process_image(path: str):
    name = Path(path).stem
    img = Image.open(path).convert("RGB")  # Ensure 3 channels
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Image {path} is not RGB (shape: {arr.shape})")

    # Конвертация в HSL и обработка канала L
    H, L, S = rgb_to_hls_arr(arr)
    L_gray = (L*255).astype(np.uint8)
    
    # Вычисление LBP и признаков для исходного изображения
    lbp_orig = compute_lbp(L_gray)
    hist_orig = compute_lbp_histogram(lbp_orig)
    entropy0, mean0 = compute_features(hist_orig)

    # Степенное преобразование
    L_trans = power_transform(L)
    L_trans_gray = (L_trans*255).astype(np.uint8)
    arr_trans = hls_to_rgb_arr(H, L_trans, S)

    # Вычисление LBP и признаков после преобразования
    lbp_trans = compute_lbp(L_trans_gray)
    hist_trans = compute_lbp_histogram(lbp_trans)
    entropy1, mean1 = compute_features(hist_trans)

    # Сохранение результатов
    Image.fromarray(L_gray).save(f"{DST_DIR}/{name}_gray.png")
    Image.fromarray(L_trans_gray).save(f"{DST_DIR}/{name}_gray_trans.png")
    
    # Визуализация гистограмм
    plt.figure(figsize=(12,6))
    plt.subplot(121); plt.hist(L_gray.ravel(), 256); plt.title("Яркость до")
    plt.subplot(122); plt.hist(L_trans_gray.ravel(), 256); plt.title("Яркость после")
    plt.savefig(f"{DST_DIR}/{name}_hists.png"); plt.close()

    # Гистограммы LBP
    plt.figure(figsize=(12,6))
    plt.subplot(121); plt.bar(range(256), hist_orig); plt.title("LBP до")
    plt.subplot(122); plt.bar(range(256), hist_trans); plt.title("LBP после")
    plt.savefig(f"{DST_DIR}/{name}_lbp_hists.png"); plt.close()

    # Сохранение признаков
    with open(f"{DST_DIR}/{name}_features.txt", "w") as f:
        f.write(f"Энтропия: {entropy0:.2f} -> {entropy1:.2f}\n")
        f.write(f"Среднее: {mean0:.2f} -> {mean1:.2f}\n")


def main():
    files = glob.glob(os.path.join(SRC_DIR, "*.*"))
    for path in files:
        try:
            process_image(path)
        except Exception as e:
            print(f"[!]{path} — ошибка: {e}")

if __name__ == "__main__":
    main()