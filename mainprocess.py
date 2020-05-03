# -- coding: utf-8 --

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from preprocessing import load_all_img, load_face_img, load_bust_img, load_portrait_img, clast2cfirst, cfirst2clast
import cv2

"""
やるべきこと
・入力画像に類似した画像をGANで生成
0.入力画像の読み込み(preprocessingにて実装済)
1.DCGANの実装
    ・実装の簡略化
    ※あとからPGGANなどに差し替えられるようにすること
2.推論のみモードへの切り替え

やりたいこと
・モノクロモードの実装
・彩度/明度の変更・左右反転による画像の水増し
    ※目的は判別でないので回転・クロップ系統の水増しは避けた方が無難
・PGGANでの実装
    ・段階的な学習による安定化
    ・計算コストの削減
"""
mode = "face"
pct_size = 100
is_channel_first = False
grayscale = True
inflate = True

def __load_img(mode="face"):
    mode = mode.lower()
    if "face" in mode:
        return load_face_img(pct_size, is_channel_first)
    elif "bust" in mode:
        return load_bust_img(pct_size, is_channel_first)
    elif "portrait" in mode:
        return load_portrait_img(pct_size, is_channel_first)
    elif "all" in mode:
        return load_all_img(pct_size, is_channel_first)
    else:
        raise ValueError('function:__load_img must need valid "mode" name')

def __bgr2rgb(img_array, arr=True):
    if not arr:
        if is_channel_first:
            return img_array[::-1,:,:]
        else:
            return img_array[:,:,::-1]
    else:
        if is_channel_first:
            return img_array[:,::-1,:,:]
        else:
            return img_array[:,:,:,::-1]

def __rgb2bgr(img_array, arr=True):
    return __bgr2rgb(img_array, arr)

def __sample_plot(img_array, id=0):
    import matplotlib.pyplot as plt

    def img_show(img, cmap='gray', vmin=0, vmax=255, interpolation='none') -> None:
        # dtypeをuint8にする
        img = np.clip(img, vmin, vmax).astype(np.uint8)
        # 画像を表示
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
        plt.show()
        plt.close()

    if id == -1:
        for i in range(img_array.shape[0]):
            img_show(__img2std_reverse(__bgr2rgb(img_array[i], arr=False)))
    else:
        img_show(__img2std_reverse(__bgr2rgb(img_array[id], arr=False)))


def __img2grayscale(std_img_array):
    if is_channel_first:
        ch = 1

    else:
        ch = 3
    sh = std_img_array.shape
    new_array = np.empty([6 * sh[0], sh[1], sh[2], sh[3]])

    img_array = __img2std_reverse(std_img_array)    # for HDTV coefficient method
    R,G,B = 2,1,0   # cv2
    mid_value_method = lambda img: np.max(img, axis=ch-1)/2 + np.min(img, axis=ch-1)/2
    ntsc_coef_method = lambda img: 0.298912*img[:, :, R] + 0.586611*img[:,:,G] + 0.114478*img[:,:,B]
    hdtv_coef_method = lambda img, x=2.2: (0.222015*(img[:, :, R]**x) + 0.706655*(img[:,:,G]**x) + 0.071330*(img[:,:,B]**x))**(1.0/x)
    simp_mean_method = lambda img: np.mean(img, axis=ch-1)
    g_channel_method = lambda img: img[:,:,G]
    med_value_method = lambda img: np.median(img, axis=ch-1)

    pseudo_gray = lambda img_gray: np.einsum('ij,k->ijk', img_gray,[1,1,1])

    for i, img in enumerate(img_array):
        new_array[i + 0*sh[0]] = pseudo_gray(mid_value_method(img))
        new_array[i + 1*sh[0]] = pseudo_gray(ntsc_coef_method(img))
        new_array[i + 2*sh[0]] = pseudo_gray(hdtv_coef_method(img))
        new_array[i + 3*sh[0]] = pseudo_gray(simp_mean_method(img))
        new_array[i + 4*sh[0]] = pseudo_gray(g_channel_method(img))
        new_array[i + 5*sh[0]] = pseudo_gray(med_value_method(img))

    std_img_array = __img2std(new_array)

    for i in range(36):
        __sample_plot(std_img_array, id=0 + i*int(sh[0]/6))

    return std_img_array

# from http://adragoona.hatenablog.com/entry/2017/12/04/010526
def __adjust_hue_saturation_lightness(
        img, hue_offset=0.0, saturation=0.0, lightness=0.0):
    # hue is mapped to [0, 1] from [0, 360]
    if hue_offset not in range(-180, 180):
        raise ValueError('Hue should be within (-180, 180)')
    if saturation not in range(-100, 100):
        raise ValueError('Saturation should be within (-100, 100)')
    if lightness not in range(-100, 100):
        raise ValueError('Lightness should be within (-100, 100)')
    img = rgb2hsv(img)
    offset = ((180 + hue_offset) % 180) / 360.0
    img[:, :, 0] = img[:, :, 0] + offset
    img[:, :, 1] = img[:, :, 1] + saturation / 200.0
    img[:, :, 2] = img[:, :, 2] + lightness / 200.0
    img = np.clip(hsv2rgb(img), 0, 255)
    return img

def __rotate_img(std_img_array, rot_variation=6, max_rot_deg=15):
    if rot_variation < 2:
        if rot_variation == 1:
            return std_img_array
        else:
            raise ValueError("rot_variation should be greater than 1")
    else:
        if max_rot_deg < 0:
            raise ValueError("max_rot_deg should be greater than 0")

        img_array = __img2std_reverse(std_img_array)

        sh = std_img_array.shape
        new_array = np.empty([rot_variation * sh[0], sh[1], sh[2], sh[3]])
        inc_rot = max_rot_deg / rot_variation
        min_deg = - max_rot_deg / 2

        for i, img in enumerate(img_array):
            new_array[i + 0 * sh[0]] = pseudo_gray(mid_value_method(img))
            new_array[i + 1 * sh[0]] = pseudo_gray(ntsc_coef_method(img))
            new_array[i + 2 * sh[0]] = pseudo_gray(hdtv_coef_method(img))
            new_array[i + 3 * sh[0]] = pseudo_gray(simp_mean_method(img))
            new_array[i + 4 * sh[0]] = pseudo_gray(g_channel_method(img))
            new_array[i + 5 * sh[0]] = pseudo_gray(med_value_method(img))


def __inflate(std_img_array, hdv_variation=6):
    if hdv_variation < 2:
        if hdv_variation == 1:
            return std_img_array
        else:
            raise ValueError("hdv_variation should be greater than 1")
    else:
        img_array_rgb = __bgr2rgb(__img2std_reverse(std_img_array))

        if is_channel_first:
            img_array_rgb = cfirst2clast(img_array_rgb)

        sh = std_img_array.shape
        new_array = np.empty([hdv_variation * sh[0], sh[1], sh[2], sh[3]])

        # saturation_range = (-40, 40)
        max_sat = 40
        inc_sat = 2*max_sat / (hdv_variation-1)
        for i, img in enumerate(img_array_rgb):
            for j in range(hdv_variation):
                if not grayscale:
                    new_array[i + j*sh[0]] = __adjust_hue_saturation_lightness(img_array_rgb[i], 0, -max_sat+j*inc_sat, 0)
                else:
                    new_array[i + j*sh[0]] = __adjust_hue_saturation_lightness(img_array_rgb[i], np.random.randint(-179,179,1), -max_sat+j*inc_sat, np.random.randint(-99,99,1))

        if is_channel_first:
            new_array = clast2cfirst(new_array)

        std_img_array = __img2std(__rgb2bgr(new_array))
        return std_img_array

def __img2std(img_array):
    return img_array / 127.5 - 1   # [-1, 1]

def __img2std_reverse(img_array):
    return (img_array + 1) * 127.5

def load_data(mode="face"):
    loaded_img = __load_img(mode)
    # 読み込んだ上で標準化
    loaded_img = __img2std(loaded_img)

    # 適宜水増し処理
    if inflate:
        loaded_img = __inflate(loaded_img)

    # グレースケール化
    if grayscale:
        loaded_img = __img2grayscale(loaded_img)

    return loaded_img

def main():
    x_train = load_data()

if __name__ == '__main__':
    main()