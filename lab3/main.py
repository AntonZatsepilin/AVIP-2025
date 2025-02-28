from PIL import Image
import numpy as np
import os
from itertools import product

BASE_DIR = "lab3"

def create_directories():
    os.makedirs(os.path.join(BASE_DIR, 'src'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'out'), exist_ok=True)

def sparse_cross_mask():
    return np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=bool)

def median_filter(image_array, mask):
    height, width = image_array.shape
    filtered = np.zeros_like(image_array)
    pad = 1 
    
    padded = np.pad(image_array, pad, mode='edge')
    
    mask_coords = [(i-1, j-1) for i, j in product(range(3), repeat=2) if mask[i,j]]
    
    for y in range(height):
        for x in range(width):
            values = [padded[y + i + pad, x + j + pad] 
                     for i, j in mask_coords]
            filtered[y, x] = sorted(values)[2]
    
    return filtered

def create_difference(original, filtered):
    return np.abs(original.astype(int) - filtered.astype(int)).astype(np.uint8)

def process_image(image_path, output_dir):
    try:
        with Image.open(image_path) as img:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if img.mode != 'L':
                img = img.convert('L')
            
            img_array = np.array(img)
            mask = sparse_cross_mask()
            
            filtered_array = median_filter(img_array, mask)
            
            filtered_img = Image.fromarray(filtered_array)
            filtered_path = os.path.join(output_dir, f"{base_name}_filtered.bmp")
            filtered_img.save(filtered_path)
            
            diff_array = create_difference(img_array, filtered_array)
            diff_img = Image.fromarray(diff_array)
            diff_path = os.path.join(output_dir, f"{base_name}_diff.bmp")
            diff_img.save(diff_path)
            
            combined = Image.new('L', (img.width*2, img.height))
            combined.paste(img, (0, 0))
            combined.paste(filtered_img, (img.width, 0))
            combined_path = os.path.join(output_dir, f"{base_name}_combined.bmp")
            combined.save(combined_path)
            
            print(f"Обработано: {os.path.basename(image_path)}")
            
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {str(e)}")

def main():
    create_directories()
    src_dir = os.path.join(BASE_DIR, 'src')
    out_dir = os.path.join(BASE_DIR, 'out')
    
    if not os.path.exists(src_dir):
        print(f"Директория с изображениями не найдена: {src_dir}")
        return
    
    processed = 0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.bmp', '.png', '.jpg')):
            process_image(os.path.join(src_dir, filename), out_dir)
            processed += 1
    
    if processed == 0:
        print("Не найдено подходящих изображений в формате .bmp/.png/.jpg")
    else:
        print(f"\nОбработка завершена. Результаты в: {out_dir}")

if __name__ == '__main__':
    main()