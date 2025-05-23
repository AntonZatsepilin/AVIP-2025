from PIL import Image, ImageDraw, ImageFont
import os

output_folder = os.path.join(os.path.dirname(__file__), "images")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

letters = "גדהוזחטיכךלמםנןסעפףצץקרשתﭏ" # hebrew alphabet

font = ImageFont.truetype("Times New Roman.ttf", 52)

for letter in letters:
    imgLen = 50
    img = Image.new('L', (imgLen, imgLen), color=255) 
    draw = ImageDraw.Draw(img)

    text_width, text_height = draw.textbbox((0, 0), letter, font=font)[2:4]
    position = ((imgLen - text_width) // 2, (imgLen - text_height - 14) // 2)

    draw.text(position, letter, fill=0, font=font)

    img.save(f"{output_folder}/{letter}.png")
