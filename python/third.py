# クロップした画像から才能開花を読み取る
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import pyocr


def main():
    for i in [1]:  # 7, 11,32
        # img = cv.imread(f"../img/item_{i}.png")
        img = cv.imread("./vertical.png")
        # plt.imshow(img[50:90, 0:45])
        # plt.show()
        # imgshow(img[49:91, 3:45])
        match(img)
        # cv.imwrite("t_4.png", img[50:90, 0:45])


def match(img):
    # 画像の読み込み + グレースケール化
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    template = cv.imread("./t_1.png")
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    # 処理対象画像に対して、テンプレート画像との類似度を算出する
    res = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)

    # 類似度の高い部分を検出する
    threshold = 0.8
    loc = np.where(res >= threshold)

    # テンプレートマッチング画像の高さ、幅を取得する
    h, w = template_gray.shape

    # 検出した部分に赤枠をつける
    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    imgshow(img)


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


if __name__ == "__main__":
    main()
