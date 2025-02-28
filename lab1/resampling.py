from PIL import Image
import numpy as np
from os import path


def image_to_np_array(image_name: str) -> np.array:
    img_src = Image.open(path.join('lab1', 'pictures_src', image_name)).convert('RGB')
    return np.array(img_src)


def two_step_resampling(img: np.array, numerator: int,
                        denominator: int) -> np.array:
    tmp = one_step_resampling(img, numerator, lambda a, b: a * b,
                              lambda a, b: int(round(a / b)))
    return one_step_resampling(tmp, denominator,
                               lambda a, b: int(round(a / b)),
                               lambda a, b: a * b)


def one_step_resampling(img: np.array, factor: float, f1, f2):
    dimensions = img.shape[0:2]
    new_dimensions = tuple(f1(dimension, factor) for dimension in
                           dimensions)
    new_shape = (*new_dimensions, img.shape[2])
    new_img = np.empty(new_shape)

    for x in range(new_dimensions[0]):
        for y in range(new_dimensions[1]):
            new_img[x, y] = img[
                min(f2(x, factor), dimensions[0] - 1),
                min(f2(y, factor), dimensions[1] - 1)
            ]
    return new_img


if __name__ == '__main__':
    pass