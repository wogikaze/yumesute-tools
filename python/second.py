import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import pyocr


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


def get_items():
    stash = Image.open("./vertical.png")
    item_size = 185
    start_x, start_y = 32, 200
    gap_x, gap_y = 16, 35
    items = []
    x, y = start_x, start_y
    while y + item_size + gap_y < stash.height:
        while x + item_size + gap_x < stash.width:
            cropped_image = stash.crop((x, y, x + item_size, y + item_size))
            items.append(cropped_image)
            x += item_size + gap_x
        y += item_size + gap_y
        x = start_x
    return items


def optimize(image):
    border = 200  # 180<<<200<= <<<220
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
    arr = cv.resize(arr, (int(arr.shape[0] * 4), int(arr.shape[1] * 4)))
    dilation = cv.dilate(arr,kernel,iterations = 1)

    imgshow(arr[153 * 4 : len(arr), 42 * 4 : 85 * 4])
    return Image.fromarray(arr[153 * 4 : len(arr), 42 * 4 : 85 * 4])


def recognize(image):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool")
    return tools[0].image_to_string(
        image,
        lang="eng",
        builder=pyocr.builders.DigitBuilder(tesseract_layout=10),
    )


items = get_items()
for item in range(2):  # 1,7,27, 40
    arr = optimize(items[item])
    text = recognize(arr)
    print(text)
