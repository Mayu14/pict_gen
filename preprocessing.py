# -- coding: utf-8 --

from pathlib import Path
import numpy as np
import cv2
import pickle

"""
やるべきこと
・入力画像サイズを指定値に変更(一斉リサイズ)(実装済)
・画像→ndarray(実装済)
・教師データとして使える形でpklとして保存(実装済)

やりたいこと
・顔検知→周辺の一定領域のみ切り出し(精度向上のため)
・同様に回転
    →顔検出＆顔の中に目を検出できる角度を”正しい”とみなして保存
"""
pct_size = 100
pct_channel = 3
is_channel_first = False    # for keras
# is_channel_first = True     # for PyTorch

def load_all_img(size=pct_size, is_c_first=is_channel_first):
    return __load_img("all", size=size, is_c_first=is_c_first)

def load_bust_img(size=pct_size, is_c_first=is_channel_first):
    return __load_img("bust", size=size, is_c_first=is_c_first)

def load_face_img(size=pct_size, is_c_first=is_channel_first):
    return __load_img("face", size=size, is_c_first=is_c_first)

def load_portrait_img(size=pct_size, is_c_first=is_channel_first):
    return __load_img("portrait", size=size, is_c_first=is_c_first)

def load_extracted_img(size=pct_size, is_c_first=is_channel_first):
    return __load_img("extracted", size=size, is_c_first=is_c_first)

def __error_msg(msg):
    try:
        raise ValueError(msg)
    except ValueError as e:
        print(e)

def __gen_pkl_name(key1, key2):
    return str(key1) + "_" + str(key2).zfill(4) + ".pkl"

def __resize_img_rect(img, size, algorithm=cv2.INTER_AREA):
    # cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST
    return cv2.resize(img, (size, size), interpolation=algorithm)

def clast2cfirst(img_array):
    return img_array.transpose([0, 3, 1, 2])

def cfirst2clast(img_array):
    return img_array.transpose([0, 2, 3, 1])


def __load_img(folder="bust", size=100, is_c_first=False):
    pkl = Path(__gen_pkl_name(folder, size))
    if not pkl.exists():
        __img2pickle(folder, size)
    with open(pkl, "rb") as f:
        img_array = pickle.load(f)

    if is_c_first:
        img_array = clast2cfirst(img_array)
    return img_array

def __img2pickle(folder="bust", size=100):
    folder = folder.lower()
    img_path = Path("data\\train")
    if folder == "all":
        pattern = "*/*"
    else:
        img_path = Path("data\\train") / Path(folder)
        pattern = "*"

    if not img_path.exists():
        __error_msg("img_path is not exists")
    else:
        # array準備
        flist = list(img_path.glob(pattern))
        number = len(flist)
        img_array = np.zeros((number, size, size, pct_channel))

        num = 0
        for path in img_path.glob(pattern):
            img = cv2.imread(str(path))
            if not all([i >= size for i in img.shape[0:2]]):
                __error_msg("input img size too small")

            elif not img.shape[0] == img.shape[1]:
                __error_msg("you can only use rectangle img")

            else:
                img_size = img.shape[0]
                if img_size != size:
                    img = __resize_img_rect(img, size)
                img_array[num, :, :, :] = img
                num += 1

        with open(__gen_pkl_name(folder, size), "wb") as f:
            pickle.dump(img_array, f)

# 特定フォルダ内のデータをndarrayに展開する
def main():
    main()


if __name__ == '__main__':
    imgs = load_portrait_img(is_c_first=False)