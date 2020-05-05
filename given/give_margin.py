# -- coding: utf-8 --
import cv2
from pathlib import Path

def mirror_padding(img_path):
    img1 = cv2.imread(img_path)
    padding_y = img1.shape[0] // 5
    padding_x = img1.shape[1] // 5
    img2 = cv2.copyMakeBorder(img1, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_REFLECT_101)
    return img2

if __name__ == '__main__':
    image_paths = Path("..\\data\\train").glob("*\\*.jpg")
    save_path = Path('..\\data\\train\\padded\\')
    save_path.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        img_name = image_path.with_suffix(".png")
        img = mirror_padding(str(image_path))
        cv2.imwrite(str(save_path / img_name.name), img)