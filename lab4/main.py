# lab4/main.py

import numpy as np
from PIL import Image
import os
import argparse

def rgb_to_grayscale(img_array):
    """Конвертация RGB в полутоновое изображение с сохранением размеров"""
    return np.dot(img_array[..., :3], [0.3, 0.59, 0.11]).astype(np.uint8)

def convolve(image, kernel):
    """Выполнение свертки изображения с заданным ядром"""
    kernel = np.array(kernel)
    pad_size = kernel.shape[0] // 2
    padded = np.pad(image, pad_size, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y+kernel.shape[0], x:x+kernel.shape[1]]
            output[y, x] = np.sum(region * kernel)
    return output

def normalize(gradient):
    """Нормализация матрицы к диапазону 0-255"""
    min_val = np.min(gradient)
    max_val = np.max(gradient)
    if max_val == min_val:
        return np.zeros_like(gradient, dtype=np.uint8)
    normalized = (gradient - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Выделение контуров оператором Шарра')
    parser.add_argument('--input', type=str, default='image.png', help='Имя входного файла')
    parser.add_argument('--threshold', type=int, default=50, help='Порог бинаризации (0-255)')
    args = parser.parse_args()

    # Создание директорий
    src_dir = os.path.join('lab4', 'src')
    out_dir = os.path.join('lab4', 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Загрузка изображения
    input_path = os.path.join(src_dir, args.input)
    if not os.path.exists(input_path):
        print(f"Изображение не найдено: {input_path}")
        return

    # Обработка изображения
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # Конвертация в полутоновое
    gray = rgb_to_grayscale(img_array)
    Image.fromarray(gray).save(os.path.join(out_dir, 'grayscale.png'))

    # Ядра Шарра
    Gx = [[3, 10, 3],
          [0, 0, 0],
          [-3, -10, -3]]
    
    Gy = [[3, 0, -3],
          [10, 0, -10],
          [3, 0, -3]]

    # Вычисление градиентов
    gx = convolve(gray, Gx)
    gy = convolve(gray, Gy)
    g = np.sqrt(gx**2 + gy**2)

    # Нормализация матриц
    gx_norm = normalize(gx)
    gy_norm = normalize(gy)
    g_norm = normalize(g)

    # Сохранение результатов
    Image.fromarray(gx_norm).save(os.path.join(out_dir, 'Gx.png'))
    Image.fromarray(gy_norm).save(os.path.join(out_dir, 'Gy.png'))
    Image.fromarray(g_norm).save(os.path.join(out_dir, 'G.png'))

    # Бинаризация
    binary = np.where(g_norm >= args.threshold, 255, 0).astype(np.uint8)
    Image.fromarray(binary).save(os.path.join(out_dir, 'G_binary.png'))

    print("Обработка завершена. Результаты сохранены в папке 'out'")

if __name__ == '__main__':
    main()