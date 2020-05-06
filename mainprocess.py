# -- coding: utf-8 --

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from preprocessing import load_all_img, load_face_img, load_bust_img, load_portrait_img, load_extracted_img, clast2cfirst, cfirst2clast, __resize_img_rect
from keras.preprocessing.image import ImageDataGenerator
from given.ellipse_blur import multi_block_blur_for_array

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
grayscale = False
inflate_hsv = True

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
    elif "extracted" in mode:
        return load_extracted_img(pct_size, is_channel_first)
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


def __img2grayscale(std_img_array, gray_variation=6):
    if gray_variation < 1:
        raise ValueError("hdv_variation should be greater than 1")
    else:
        if is_channel_first:
            ch = 1
        else:
            ch = 3
        sh = std_img_array.shape
        new_array = np.empty([gray_variation * sh[0], sh[1], sh[2], sh[3]])

        img_array = __img2std_reverse(std_img_array)    # for HDTV coefficient method
        R,G,B = 2,1,0   # cv2
        mid_value = lambda img: np.max(img, axis=ch-1)/2 + np.min(img, axis=ch-1)/2
        ntsc_coef = lambda img: 0.298912*img[:, :, R] + 0.586611*img[:,:,G] + 0.114478*img[:,:,B]
        hdtv_coef = lambda img, x=2.2: (0.222015*(img[:, :, R]**x) + 0.706655*(img[:,:,G]**x) + 0.071330*(img[:,:,B]**x))**(1.0/x)
        simp_mean = lambda img: np.mean(img, axis=ch-1)
        g_channel = lambda img: img[:,:,G]
        med_value = lambda img: np.median(img, axis=ch-1)
        methods = [ntsc_coef, g_channel, med_value, hdtv_coef, simp_mean, mid_value]

        pseudo_gray = lambda img_gray: np.einsum('ij,k->ijk', img_gray,[1,1,1])

        for i, img in enumerate(img_array):
            for j in range(gray_variation):
                new_array[i + j*sh[0]] = np.clip(pseudo_gray(methods[j](img)), 0, 255)

        std_img_array = __img2std(new_array)

        debug = False
        if debug:
            for i in range(36):
                __sample_plot(std_img_array, id=0 + i*int(sh[0]/6))

        return std_img_array

def __img2std(img_array):
    return img_array / 127.5 - 1   # [-1, 1]

def __img2std_reverse(img_array):
    return (img_array + 1) * 127.5

# from http://aidiary.hatenablog.com/entry/20161212/1481549365
def __draw_images(datagen, x, result_images):
    import os, glob, shutil
    import matplotlib.pyplot as plt
    import cv2
    from matplotlib.gridspec import GridSpec
    # 出力先ディレクトリを作成
    temp_dir = "temp"
    os.mkdir(temp_dir)

    # generatorから9個の画像を生成
    # xは1サンプルのみなのでbatch_sizeは1で固定
    x = __bgr2rgb(x)
    g = datagen.flow(x, batch_size=32, save_to_dir=temp_dir, save_prefix='img', save_format='jpg')
    for i in range(9):
        batch = g.next()
    # 生成した画像を3x3で描画
    images = glob.glob(os.path.join(temp_dir, "*.jpg"))
    fig = plt.figure()
    gs = GridSpec(3, 3)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(9):
        img = cv2.imread(images[i])
        plt.subplot(gs[i])
        plt.imshow(img, aspect='auto')
        plt.axis("off")
    plt.savefig(result_images)
    # 出力先ディレクトリを削除
    shutil.rmtree(temp_dir)

def __image_data_generator(img_array):
    datagen = ImageDataGenerator(
        # zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.025,
        height_shift_range=0.025,
        shear_range=0.05,
        zoom_range=[0.98,1.02],
        channel_shift_range=0.05,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False
    )
    debug = False
    if debug:
        __draw_images(datagen,img_array,"result.jpg")
    return datagen

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

def __inflate_hsv(std_img_array, hdv_variation=6):
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
                    new_array[i + j*sh[0]] = __adjust_hue_saturation_lightness(img_array_rgb[i], 0, int(-max_sat+j*inc_sat), 0)
                else:
                    # new_array[i + j*sh[0]] = __adjust_hue_saturation_lightness(img_array_rgb[i], np.random.randint(-179,179,1), -max_sat+j*inc_sat, np.random.randint(-99,99,1))
                    new_array[i + j * sh[0]] = __adjust_hue_saturation_lightness(img_array_rgb[i], 0, int(-max_sat + j * inc_sat), 0)

        if is_channel_first:
            new_array = clast2cfirst(new_array)

        std_img_array = __img2std(__rgb2bgr(new_array))
        return std_img_array


def load_data(mode="face", size=-1):
    loaded_img = __load_img(mode)
    #loaded_img = multi_block_blur_for_array(loaded_img)
    # 読み込んだ上で標準化
    loaded_img = __img2std(loaded_img)
    ch = 3
    # 適宜水増し処理
    if inflate_hsv:
        loaded_img = __inflate_hsv(loaded_img, hdv_variation=8)
    # グレースケール化
    if grayscale:
        ch = 1
        loaded_img = __img2grayscale(loaded_img)
        loaded_img = np.max(loaded_img, axis=3, keepdims=True)

    if size > 0:
        import cv2
        original_img = loaded_img
        sh = loaded_img.shape
        if is_channel_first:
            loaded_img = np.empty((2*sh[0], sh[1], size, size))
            reshape = (ch, size, size)
        else:
            loaded_img = np.empty((2*sh[0], size, size, sh[3]))
            reshape = (size, size, ch)
        for i, img in enumerate(original_img):
            loaded_img[i] = __resize_img_rect(img, size, algorithm=cv2.INTER_LANCZOS4).reshape(reshape)
            loaded_img[i + sh[0]] = __resize_img_rect(img, size, algorithm=cv2.INTER_AREA).reshape(reshape)

    datagen = __image_data_generator(loaded_img)

    return loaded_img, datagen

def main():
    x_train, _ = load_data()
    print(x_train.shape)
    print(np.max(x_train), np.min(x_train))


if __name__ == '__main__':
    main()



















