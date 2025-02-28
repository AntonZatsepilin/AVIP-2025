from PIL import Image
import numpy as np
import os
import math

BASE_DIR = "lab2"

def create_directories():
    os.makedirs(os.path.join(BASE_DIR, 'src'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'out'), exist_ok=True)

def to_greyscale_balanced(image):
    data = image.load()
    result = Image.new('L', image.size)
    new_data = result.load()
    
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r, g, b = data[x, y]
            new_data[x, y] = int(0.3 * r + 0.59 * g + 0.11 * b)
    return result

def eikvel_binarization(image_array, diff=5, inner=3, outer=15):
    height, width = image_array.shape
    new_image = np.zeros_like(image_array)
    half_outer = outer // 2
    
    for y in range(0, height, inner):
        direction = 1 if (y // inner) % 2 == 0 else -1
        
        x_range = range(0, width, inner) if direction == 1 else range(width-1, -1, -inner)
        
        for x in x_range:
            y_start = max(0, y - half_outer)
            y_end = min(height, y + half_outer + inner)
            x_start = max(0, x - half_outer)
            x_end = min(width, x + half_outer + inner)
            
            large_window = image_array[y_start:y_end, x_start:x_end]
            if large_window.size == 0:
                continue
                
            small_y_start = y
            small_y_end = min(height, y + inner)
            small_x_start = x
            small_x_end = min(width, x + inner)
            small_window = image_array[small_y_start:small_y_end, small_x_start:small_x_end]
            
            threshold = np.mean(large_window)
            
            mask = small_window > threshold
            new_image[small_y_start:small_y_end, small_x_start:small_x_end][mask] = 255
            
    return new_image.astype(np.uint8)

def process_image(input_path, output_dir):

    try:
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            grey_img = to_greyscale_balanced(img)
            grey_path = os.path.join(output_dir, f"{base_name}_grey.bmp")
            grey_img.save(grey_path)
            
            binary_array = eikvel_binarization(np.array(grey_img))
            binary_img = Image.fromarray(binary_array, 'L')
            binary_path = os.path.join(output_dir, f"{base_name}_binary.bmp")
            binary_img.save(binary_path)
            
            print(f"Успешно обработан: {os.path.basename(input_path)}")
    except Exception as e:
        print(f"Ошибка обработки {input_path}: {str(e)}")

if __name__ == '__main__':
    create_directories()
    
    src_dir = os.path.join(BASE_DIR, 'src')
    out_dir = os.path.join(BASE_DIR, 'out')
    
    if not os.path.exists(src_dir):
        print(f"Директория с исходными изображениями не найдена: {src_dir}")
        exit(1)
    
    processed = 0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.bmp', '.png')):
            input_path = os.path.join(src_dir, filename)
            process_image(input_path, out_dir)
            processed += 1
    
    if processed == 0:
        print("Не найдено изображений для обработки в формате .bmp или .png")
        print(f"Проверьте папку: {os.path.abspath(src_dir)}")
    else:
        print(f"\nОбработка завершена. Результаты сохранены в: {os.path.abspath(out_dir)}")