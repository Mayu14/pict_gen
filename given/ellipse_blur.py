import cv2
import numpy as np

# 外部のみにぼかし適用
def apply_ellipse_blur(image, x, y, haxis, vaxis, angle, ksize):
    # 全体をぼかす
    blurred_image = image.copy()
    blurred_image = cv2.blur(blurred_image, ksize)
    # 楕円形マスクの作成
    maskShape = (image.shape[0], image.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)
    cv2.ellipse(mask, ((x, y), (haxis, vaxis), angle), (255), -1)
    # 楕円形マスク外部を取り出す
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_inv)
    # 楕円形マスク内部を取り出す
    img2_fg = cv2.bitwise_and(image, image, mask=mask)
    # 合成する
    return cv2.add(img1_bg, img2_fg)

def multi_block_blur(img):
    width, height = img.shape[:2]
    cx, cy = int(width/2), int(height/2)
    # 90%, 95%, 100%地点で多段ぼかし
    r_pers = [0.9, 0.95, 1.0]

    for r_per in r_pers:
        r = int(r_per*cx)
        img = apply_ellipse_blur(img, cx, cy, r, r, 0, ksize=(3,3))
    return img

def example():
    image = cv2.imread('pic.jpg')
    # ぼかす中心座標
    x = 256
    y = 256
    # 楕円横方向径
    haxis = 250
    # 楕円縦方向径
    vaxis = 250
    # 楕円傾き
    angle = 20
    # ぼけカーネルサイズ
    ksize = (15, 15)
    blurred_image = apply_ellipse_blur(image, x, y, haxis, vaxis, angle, ksize)
    cv2.imwrite('blurred_pic.png', blurred_image)


if __name__ == '__main__':
    image = cv2.imread('pic.jpg')
    multi_block_blur()
