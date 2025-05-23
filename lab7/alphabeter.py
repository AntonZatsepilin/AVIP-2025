import os
import numpy as np
from PIL import Image, ImageDraw

base_dir = os.path.dirname(__file__)
SRC_PATH = os.path.join(base_dir, "mphrase.bmp")
DST_DIR = os.path.join(base_dir, "alphabet")

alphabet = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת'

os.makedirs(DST_DIR, exist_ok=True)

def binarize_image(grayscale_image, threshold=128):
    binary_image = grayscale_image < threshold
    return binary_image.astype(np.uint8)

def to_binary(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    arr = binarize_image(arr, 130)
    char_img = Image.fromarray((1 - arr) * 255)
    char_img.save(os.path.join(DST_DIR, "binarizated.bmp"))
    return arr

def profiles(bin_img: np.ndarray):
    """Возвращает (горизонтальный, вертикальный) профили."""
    return bin_img.sum(axis=1), bin_img.sum(axis=0)


def segment_by_profiles(bin_img: np.ndarray, empty_thresh: int = 2):
    """
    Разбивает строку на буквы по вертикальному профилю.
    empty_thresh — сколько чёрных пикселей в столбце считаем «пустым».
    """
    h, w = bin_img.shape 
    vert = bin_img.sum(axis=0)
    splits, in_char = [], False

    for x, v in enumerate(vert):
        if not in_char and v > empty_thresh:
            in_char = True
            x0 = x
        elif in_char and v <= empty_thresh:
            splits.append((x0, x-1))
            in_char = False
    if in_char:
        splits.append((x0, w-1))

    boxes = []
    for x0, x1 in splits:
        slice_ = bin_img[:, x0:x1+1]
        horiz = slice_.sum(axis=1)
        ys = np.where(horiz > empty_thresh)[0]
        if ys.size:
            boxes.append((x0, ys[0], x1, ys[-1]))
    return boxes

def split_wide_boxes(boxes: list[tuple], bin_img: np.ndarray, factor: float = 1.4):
    """
    Если бокс шире, чем factor * средняя ширина всех боксов,
    разрезаем его по локальному минимуму вертикального профиля.
    """
    widths = [x1 - x0 + 1 for x0,_,x1,_ in boxes]
    if not widths:
        return boxes
    avg_w = sum(widths) / len(widths)

    out = []
    for (x0,y0,x1,y1), w in zip(boxes, widths):
        if w > avg_w * factor:
            sub = bin_img[y0:y1+1, x0:x1+1]
            vert = sub.sum(axis=0)
            margin = max(w // 10, 1)
            local = vert[margin:-margin]
            if local.size:
                cut_rel = int(np.argmin(local)) + margin
                cut = x0 + cut_rel
                out += [(x0,y0,cut,y1), (cut+1,y0,x1,y1)]
                continue
        out.append((x0,y0,x1,y1))
    return out


def save_letter_profiles(bin_img: np.ndarray, boxes: list[tuple]):
    for idx, (x0,y0,x1,y1) in enumerate(boxes):
        patch = bin_img[y0:y1+1, x0:x1+1]
        char_img = Image.fromarray((1 - patch) * 255)
        bmp_name = f"{alphabet[idx]}.bmp" 
        char_img.save(os.path.join(DST_DIR, bmp_name))

def draw_boxes(path: str, boxes: list[tuple]):
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle([ (x0,y0), (x1,y1) ], outline="red", width=2)
    img.save(os.path.join(DST_DIR, "phrase_boxes_fixed.bmp"))

bin_img = to_binary(SRC_PATH)


boxes = segment_by_profiles(bin_img, empty_thresh=2)
boxes = split_wide_boxes(boxes, bin_img, factor=2)
boxes.reverse()  

draw_boxes(SRC_PATH, boxes)
save_letter_profiles(bin_img, boxes)

print(f"Сегментировано символов: {len(boxes)}")
print(f"Результаты в {DST_DIR}")