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
    pad_size = 1 
    
    padded = np.pad(image_array, pad_size, mode='edge')
    
    mask_coords = [(i, j) for i, j in product(range(3), range(3)) if mask[i,j]]
    
    for y in range(height):
        for x in range(width):
            values = [padded[y + i, x + j] for i, j in mask_coords]
            filtered[y, x] = sorted(values)[2]
    
    return filtered

def apply_to_color(img_color, mask):
    r, g, b = img_color.split()
    
    r_filtered = median_filter(np.array(r), mask)
    g_filtered = median_filter(np.array(g), mask)
    b_filtered = median_filter(np.array(b), mask)
    
    return Image.merge('RGB', (
        Image.fromarray(r_filtered),
        Image.fromarray(g_filtered),
        Image.fromarray(b_filtered)
    ))

def create_difference(original, filtered):
    return Image.fromarray(
        np.abs(original.astype(int) - filtered.astype(int)).astype(np.uint8)
    )

def process_image(image_path, output_dir):
    try:
        with Image.open(image_path) as img:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask = sparse_cross_mask()
            
            if img.mode == 'L':
                grey_array = np.array(img)
                filtered_array = median_filter(grey_array, mask)
                filtered_img = Image.fromarray(filtered_array)
            
                diff_img = create_difference(grey_array, filtered_array)
                
                filtered_img.save(os.path.join(output_dir, f"{base_name}_filtered.bmp"))
                diff_img.save(os.path.join(output_dir, f"{base_name}_diff.bmp"))
                
            else:
                grey_img = img.convert('L')
                grey_array = np.array(grey_img)
                filtered_grey = median_filter(grey_array, mask)
                
                color_filtered = apply_to_color(img, mask)
                
                grey_img.save(os.path.join(output_dir, f"{base_name}_grey.bmp"))
                Image.fromarray(filtered_grey).save(os.path.join(output_dir, f"{base_name}_grey_filtered.bmp"))
                color_filtered.save(os.path.join(output_dir, f"{base_name}_color_filtered.bmp"))

                create_difference(grey_array, filtered_grey).save(
                    os.path.join(output_dir, f"{base_name}_grey_diff.bmp"))
                
            print(f"Обработано: {base_name}")
            
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {str(e)}")

def main():
    create_directories()
    src_dir = os.path.join(BASE_DIR, 'src')
    out_dir = os.path.join(BASE_DIR, 'out')
    
    if not os.path.exists(src_dir):
        print(f"Директория не найдена: {src_dir}")
        return
    
    processed = 0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.bmp', '.png')): 
            process_image(os.path.join(src_dir, filename), out_dir)
            processed += 1
    
    if processed == 0:
        print("Не найдено изображений .bmp/.png в папке src")
    else:
        print(f"Обработано изображений: {processed}")
        print(f"Результаты в: {os.path.abspath(out_dir)}")

if __name__ == '__main__':
    main()