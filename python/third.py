# クロップした画像を読み取る
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pyocr


def main():
    for i in range(40, 54):  # 7, 11,32
        img = cv2.imread(f"../img/item_{i}.png")

        img_arr = optimize(img)
        dil_img = cv2.dilate(img_arr, np.ones((2, 2)), iterations=1)
        ero_img = cv2.erode(dil_img, np.ones((2, 2)), iterations=1)
        # imgshow(ero_img)

        text = recognize(Image.fromarray(ero_img))
        print(text)


def recognize(image):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool")
    return tools[0].image_to_string(
        image,
        lang="eng",
        builder=pyocr.builders.DigitBuilder(tesseract_layout=10),
    )


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
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # BGRA
            elif img.shape[2] == 4:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    # 1枚
    else:
        plt.figure(figsize=(8, 16))
        if src.ndim == 2:
            plt.imshow(src, cmap="gray", vmin=0, vmax=255)
        elif src.shape[2] == 3:
            plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        elif src.shape[2] == 4:
            plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))

    plt.show()


def optimize(image):
    border = 190  # 180<<<200<= <<<220
    arr = np.array(image)

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            pix = arr[i][j]
            if j < 42 or i < 153 or j > 80:  # 数字以外の座標
                arr[i][j] = [0, 0, 0]
            elif pix[0] < border or pix[1] < border or pix[2] < border:
                arr[i][j] = [0, 0, 0]
            elif (
                pix[0] >= border or pix[1] >= border or pix[2] >= border
            ):  # 白文字は黒に
                arr[i][j] = [255, 255, 255]
    # imgshow(arr)
    return arr


if __name__ == "__main__":
    main()
