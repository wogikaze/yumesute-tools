# クロップした画像から才能開花を読み取る
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image

def main():
    for i in [4]:  # 7, 11,32
        img = cv.imread(f"img/item_{i}.png")    #アクターアイコン
        
        ft = match(img)     #才能開花の数を返す
        print(ft)

def match(img):
    best_match = None  # 最も類似度の高いテンプレート番号を保存する変数
    best_score = 0.0   # 最も類似度の高いスコアを保存する変数

    for i in range(6): #0,1,2,3,4,5
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #グレイスケール

        template = cv.imread(f"img/t_{i}.png", 0)  # テンプレート画像をグレースケールで読み込み
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

        max_score = np.max(res)
        print(max_score)
        if max_score > best_score:
            best_match = i
            best_score = max_score

    return best_match

def make_mask(template):
    # 全て白のマスクを作成
    mask = np.ones(template.shape, dtype=np.uint8) * 255

    # 中心を原点とした座標系を作成
    center = np.array(template.shape) / 2.0
    Y, X = np.indices(template.shape)
   # 中心からのマンハッタン距離を計算
    manhattan_distance = np.abs(X - center[1]) + np.abs(Y - center[0])

    # sizeを計算
    size = template.shape[0] / np.sqrt(2)**2

    # |x| + |y| = size の範囲外を黒にする
    mask[manhattan_distance > size] = 0
    # imgshow(mask)
    return mask

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
