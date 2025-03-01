from resampling import *
import os
from color import *

def prompt(variants: dict):
    for number, variant in enumerate(variants.keys(), 1):
        print(f'{number} - {variant}')

    input_correct = False
    user_input = 0

    while not input_correct:
        try:
            user_input = int(input('> '))
            if user_input <= 0 or user_input > len(variants):
                raise ValueError
            input_correct = True
        except ValueError:
            print("Введите корректное значение")

    return dict(enumerate(variants.values(), 1))[user_input]


def safe_number_input(number_type: type, lower_bound=None, upper_bound=None):
    input_correct = False
    user_input = 0

    while not input_correct:
        try:
            user_input = number_type(input('> '))
            if lower_bound is not None and user_input < lower_bound:
                raise ValueError
            if upper_bound is not None and user_input > upper_bound:
                raise ValueError
            input_correct = True
        except ValueError:
            print("Введите корректное значение")
    return user_input


def execute(img, f1, f2, number_type=int):
    data_type = np.uint8
    color_model = 'RGB'

    factor = safe_number_input(number_type, 0.5)
    result = Image.fromarray(one_step_resampling(
        img, factor, f1, f2).astype(data_type), color_model)

    return result


if __name__ == '__main__':
    images = {
        'Pokemon': 'pokemon.png'
    }
    
    operation_categories = {
        'Цветовые модели': 'color',
        'Передискретизация': 'resample'
    }
    
    color_operations = {
        'Выделить R, G, B компоненты': 'rgb',
        'Конвертировать в HSI и сохранить яркость': 'hsi',
        'Инвертировать яркость': 'invert'
    }
    
    resample_operations = {
        'Интерполяция': 'int',
        'Децимация': 'dec',
        'Двухпроходная передискретизация': 'two',
        'Однопроходная передискретизация': 'one'
    }

    print('Выберите изображение:')
    selected_image = prompt(images)
    img = image_to_np_array(selected_image)

    print('Выберите категорию:')
    selected_category = prompt(operation_categories)

    output_dir = path.join('lab1', 'pictures_results')
    os.makedirs(output_dir, exist_ok=True)

    if selected_category == 'color':
        print('Выберите операцию:')
        selected_operation = prompt(color_operations)
        
        if selected_operation == 'rgb':
            r, g, b = split_rgb_components(img)
            Image.fromarray(r, 'RGB').save(path.join(output_dir, 'R.png'))
            Image.fromarray(g, 'RGB').save(path.join(output_dir, 'G.png'))
            Image.fromarray(b, 'RGB').save(path.join(output_dir, 'B.png'))
            print("Компоненты сохранены как R.png, G.png, B.png")
        
        elif selected_operation == 'hsi':
            hsi_img = rgb_to_hsi(img)
            i_component = (hsi_img[:, :, 2] * 255).astype(np.uint8)
            Image.fromarray(i_component, 'L').save(path.join(output_dir, 'I.png'))
            print("Яркостная компонента сохранена как I.png")
        
        elif selected_operation == 'invert':
            inverted_img = invert_intensity(img)
            Image.fromarray(inverted_img, 'RGB').save(path.join(output_dir, 'inverted.png'))
            print("Инвертированное изображение сохранено как inverted.png")

    elif selected_category == 'resample':
        print('Выберите операцию:')
        selected_operation = prompt(resample_operations)
        
        if selected_operation == 'int':
            print('Введите целый коэффициент растяжения')
            result = execute(img, lambda a, b: a * b, lambda a, b: int(round(a / b)))

        elif selected_operation == 'dec':
            print('Введите целый коэффициент сжатия')
            result = execute(img, lambda a, b: int(round(a / b)), lambda a, b: a * b)

        elif selected_operation == 'two':
            print('Введите целый коэффициент растяжения')
            numerator = safe_number_input(int, 1)
            print('Введите целый коэффициент сжатия')
            denominator = safe_number_input(int, 1)
            result = Image.fromarray(two_step_resampling(img, numerator, denominator).astype(np.uint8), 'RGB')

        elif selected_operation == 'one':
            print('Введите дробный коэффициент растяжения/сжатия')
            result = execute(img, lambda a, b: int(round(a * b)), lambda a, b: int(round(a / b)), float)

        print('Введите название для сохранения (без расширения):')
        selected_path = input().strip()
        if selected_path:
            selected_path += '.png'
            result.save(path.join(output_dir, selected_path))
            print(f"Изображение сохранено как {selected_path}")