# -- coding: utf-8 --

import numpy as np
from pathlib import Path
import cv2

head = "xml\\haarcascade_"
tail = ".xml"

# 5,6は使わない
xml_fnames = ["frontalface_default",
              "frontalface_alt",
             "frontalface_alt2",
             "frontalface_alt_tree",
             "profileface",
             "smile"
             ]

# 基本的に1以外使わない
xml_enames = ["eye",
              "eye_tree_eyeglasses",
              "lefteye_2splits",
              "righteye_2splits"
              ]

xml_ubody = Path(head + "upperbody" + tail)
xmls_eyes = []
xmls_faces = []

for xmle in xml_enames:
    xmls_eyes.append(Path(head + xmle + tail))

for xmlf in xml_fnames:
    xmls_faces.append(Path(head + xmlf + tail))

def img_loader(img_path=Path("data\\train\\portrait")):
    if not img_path.exists():
        raise ValueError("img_path is not exists")
    else:
        # array準備
        pattern = "*"
        num = 0
        for path in img_path.glob(pattern):
            yield path

def main(imgs_path=Path("data\\train\\portrait"), save_path=Path("data\\train\\processed")):
    upscaled, extended, total, cannot_detected = 0, 0, 0, 0
    save_path.mkdir(parents=True, exist_ok=True)
    error_path = save_path / Path("error")

    loader = img_loader(imgs_path)
    inc = 1.0  # 角度の変化量
    for img_path in loader:
        img = cv2.imread(str(img_path))
        total += 1
        # img = cv2.imread("data\\train\\portrait\\aizawa_yurina_003.jpg")
        height, width = img.shape[:2]
        if img.shape[0] < 300:
            img = cv2.resize(img, (int(500/width)*width, int(500/height)*height), interpolation=cv2.INTER_LANCZOS4)
            print("upscale img")
            upscaled += 1

        org = img
        height, width = img.shape[:2]
        for j in range(4):  # face_cascade用のループ
            face_cascade = cv2.CascadeClassifier(str(xmls_faces[j].absolute()))
            for i in range(int(180 / inc)): # 左右に振りながら180度回す(正味の角度刻みは2*inc)
                # 画像回転
                rotM = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=i * inc * (-1)**i, scale=1.0)
                img = cv2.warpAffine(org, rotM, dsize=(width, height), borderMode=cv2.BORDER_REPLICATE)
                # 顔検出
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray)
                # 顔を認識できたとき以下のループに入る(それ以外は角度変更してやり直し)
                for (x,y,w,h) in faces:
                    # 顔の周囲200%を切り出す関数
                    bottom = lambda x, d: int(x - d / 2)
                    top = lambda x, d: int(x + 3 * d / 2)
                    bx,by,tx,ty = bottom(x,w), bottom(y,h), top(x,w), top(y,h)
                    # 顔の周囲200%が画面外にはみ出すときは画像を拡張してやり直し
                    if (bx < 0) or (by < 0) or (tx >= width) or (ty >= height):
                        t_xy = lambda x, y: np.array([[1, 0, x * 0.5 * width], [0, 1, y * 0.5 * height]],dtype=np.float32)
                        img_lb = cv2.warpAffine(org, t_xy(1, 1), dsize=(width, height), borderMode=cv2.BORDER_REPLICATE)
                        img_lt = cv2.warpAffine(org, t_xy(1, -1), dsize=(width, height), borderMode=cv2.BORDER_REPLICATE)
                        img_rb = cv2.warpAffine(org, t_xy(-1, 1), dsize=(width, height), borderMode=cv2.BORDER_REPLICATE)
                        img_rt = cv2.warpAffine(org, t_xy(-1, -1), dsize=(width, height), borderMode=cv2.BORDER_REPLICATE)
                        org = cv2.hconcat([cv2.vconcat([img_lb, img_lt]), cv2.vconcat([img_rb, img_rt])])
                        width *= 2
                        height *= 2
                        print("extend img")
                        extended += 1
                        continue

                    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)    # 枠描画
                    # roi_gray = gray[bottom(y,h):top(y,h), bottom(x,w):top(x,w)]
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]

                    # 顔の内部のみで瞳検出
                    for k in range(1):  # 基本的にhaarcascade_eye.xmlしか使わない
                        eye_cascade = cv2.CascadeClassifier(str(xmls_eyes[k].absolute()))
                        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.11, minNeighbors=5, minSize=(1,1))
                        # 瞳が検出できたとき以下のループに入る．このとき正面を向いた顔を認識できたと定義する
                        for (ex,ey,ew,eh) in eyes:
                            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  # 枠描画
                            # cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                            new_img = img[by:ty,bx:tx]
                            fname = img_path.with_suffix(".png").name
                            cv2.imwrite(str(save_path / fname), new_img)
                            if True:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            print("cannot detect")
            error_path.mkdir(parents=True, exist_ok=True)
            fname = img_path.with_suffix(".png").name
            cv2.imwrite(str(error_path / fname), org)
            cannot_detected += 1
            continue

        print(total, upscaled, extended, cannot_detected)

if __name__ == '__main__':
    imgs_path = Path("data\\train\\portrait")
    save_path = Path("data\\train\\processed")
    main(imgs_path, save_path)