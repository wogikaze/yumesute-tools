import collections
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from enum import IntEnum, auto


# 画像表示関数
def imgshow(src):
    # 複数枚
    if type(src) is list:
        fig = plt.figure(figsize=(8 * len(src), 16))
        for i, img in enumerate(src, 1):
            ax = fig.add_subplot(1, len(src), i)
            # グレースケール
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            # BGR
            elif img.shape[2] == 3:
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            # BGRA
            elif img.shape[2] == 4:
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
    # 1枚
    else:
        plt.figure(figsize=(8, 16))
        if src.ndim == 2:
            plt.imshow(src, cmap="gray", vmin=0, vmax=255)
        elif src.shape[2] == 3:
            plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
        elif src.shape[2] == 4:
            plt.imshow(cv.cvtColor(src, cv.COLOR_BGRA2RGBA))

    plt.show()


# 画像結合関数
class SimpleMergeLayout(IntEnum):
    Vertical = auto()
    Horizontal = auto()


def simple_merge(images: list, layout: SimpleMergeLayout):
    x = 0
    y = 0

    if (layout != SimpleMergeLayout.Vertical) & (
        layout != SimpleMergeLayout.Horizontal
    ):
        return

    if layout == SimpleMergeLayout.Vertical:
        # 縦結合の場合
        width = max(m.shape[1] for m in images)
        height = sum(m.shape[0] for m in images)
    elif layout == SimpleMergeLayout.Horizontal:
        # 縦結合の場合
        width = sum(m.shape[1] for m in images)
        height = max(m.shape[0] for m in images)

    output_img = np.zeros((height, width, 4), dtype=np.uint8)
    for img in images:
        # 入力画像がBGRAになるよう色空間を変更
        # グレースケール
        if img.ndim == 2:
            merge_img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
        # BGR
        elif img.shape[2] == 3:
            merge_img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        # BGRA
        elif img.shape[2] == 4:
            merge_img = img

        if layout == SimpleMergeLayout.Vertical:
            # 縦結合の場合
            output_img[
                y : y + merge_img.shape[0], 0 : merge_img.shape[1]
            ] = merge_img
            y += merge_img.shape[0]
        elif layout == SimpleMergeLayout.Horizontal:
            # 横結合の場合
            output_img[
                0 : merge_img.shape[0], x : x + merge_img.shape[1]
            ] = merge_img
            x += merge_img.shape[1]
    return output_img


# 結合対象画像をlistへ格納
images = []

for i in np.arange(3):
    img = cv.imread(f"../img/{i + 1}.jpg", cv.IMREAD_COLOR)
    print(f"{i + 1}枚目 : {img.shape[1]}px x {img.shape[0]}px")
    if i > 0:
        # 2枚目以降は解像度違ったらエラー
        if (images[i - 1].shape[0] != img.shape[0]) | (
            images[i - 1].shape[1] != img.shape[1]
        ):
            raise Exception("異なる解像度の画像が入力されています")

    images.append(img)

img = images[0]
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
# imgshow(hsv)
min = (0, 0, 0)
max = (255, 255, 220)
inrange = cv.inRange(hsv, min, max)
imgshow(inrange)
