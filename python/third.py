# クロップした画像から才能開花を読み取る
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import pyocr


def main():
    for i in range(40, 54):  # 7, 11,32
        img = cv.imread(f"../img/item_{i}.png")


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
