from PIL import Image
import os
letters = "גדהוזחטיכךלמםנןסעפףצץקרשתﭏ" # hebrew alphabet
images_dir = 'images'

base_dir = os.path.dirname(__file__)
images_dir = os.path.join(base_dir, 'images')
inverse_dir = os.path.join(base_dir, 'inverse')
os.makedirs(inverse_dir, exist_ok=True)

def invert_image(image_path, save_path):
    img = Image.open(image_path)

    img = img.convert('L')

    img = Image.eval(img, lambda x: 255 - x)

    img.save(save_path)


for symbol in letters:
    image_path = os.path.join(images_dir, f'{symbol}.png')
    if os.path.exists(image_path):
        save_path = os.path.join(inverse_dir, f'{symbol}.png')  # исправлено
        invert_image(image_path, save_path)