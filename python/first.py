# 中央の部分を取り出し結合する
import collections
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


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


BLACK = 0
WHITE = 255
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
# imgshow(images)
img = images[0]


def getleft(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    # imgshow(hsv)
    min = (0, 0, 0)
    max = (255, 255, 225)
    inrange = cv.inRange(hsv, min, max)
    # imgshow(inrange)

    # 左端の座標を保存するリスト
    positions = []

    # 画像の幅と高さ
    h, w = inrange.shape

    # 元画像(検証用)
    tmp_img = cv.cvtColor(inrange, cv.COLOR_GRAY2BGR)

    # 上から順に見ていく
    for y in np.arange(int(h * 0.15), int(h * 0.8)):
        findBlack = False
        # 左(一番左は飛ばす)から順に見ていく
        for x in np.arange(int(w * 0.2), int(w * 0.3)):
            # Windows版ではWindow枠があると、即黒を検知してしまうので、
            # 「一度白を見つけた後に、黒を見つけたら」という条件にする
            if (not findBlack) & (inrange[y, x] == BLACK):
                findBlack = True
            elif (findBlack) & (inrange[y, x] == WHITE):
                positions.append(x)
                # 元画像(検証用)に検出点を赤で色付け
                cv.line(
                    tmp_img,
                    pt1=(x, y),
                    pt2=(x, y),
                    color=(0, 0, 255),
                    thickness=3,
                )
                break

    c = collections.Counter(positions)

    # リストから一番多い出現回数の値(座標)を取得
    margin_left = c.most_common(1)[0][0]
    # imgshow(tmp_img)
    return margin_left


def getright(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    # imgshow(hsv)
    min = (0, 0, 0)
    max = (255, 255, 225)
    inrange = cv.inRange(hsv, min, max)
    # imgshow(inrange)

    # imgshow(tmp_img)
    # 左端の座標を保存するリスト
    positions = []

    # 画像の幅と高さ
    h, w = inrange.shape

    # 元画像(検証用)
    tmp_img = cv.cvtColor(inrange, cv.COLOR_GRAY2BGR)

    # 上から順に見ていく
    for y in np.arange(int(h * 0.15), int(h * 0.8)):
        findBlack = False
        # 左(一番左は飛ばす)から順に見ていく
        for x in np.arange(int(w * 0.95), int(w * 0.9), -1):
            # Windows版ではWindow枠があると、即黒を検知してしまうので、
            # 「一度白を見つけた後に、黒を見つけたら」という条件にする
            if (not findBlack) & (inrange[y, x] == BLACK):
                findBlack = True
            elif (findBlack) & (inrange[y, x] == WHITE):
                positions.append(x)
                # 元画像(検証用)に検出点を赤で色付け
                cv.line(
                    tmp_img,
                    pt1=(x, y),
                    pt2=(x, y),
                    color=(0, 0, 255),
                    thickness=3,
                )
                break

    c = collections.Counter(positions)

    # リストから一番多い出現回数の値(座標)を取得
    margin_right = c.most_common(1)[0][0]
    # imgshow(tmp_img)
    return margin_right


def getbelow(img, margin_left, margin_right):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    # imgshow(hsv)
    min = (0, 0, 0)
    max = (255, 255, 225)
    inrange = cv.inRange(hsv, min, max)

    # 元画像(検証用)
    tmp_img = cv.cvtColor(inrange, cv.COLOR_GRAY2BGR)
    # 画像の幅と高さ
    h, w = inrange.shape
    positions = []
    for x in np.arange(margin_left, margin_right):
        for y in np.arange(int(h * 0.95), int(h * 0.71), -1):
            # 左(一番左は飛ばす)から順に見ていく
            # Windows版ではWindow枠があると、即黒を検知してしまうので、
            # 「一度白を見つけた後に、黒を見つけたら」という条件にする
            if inrange[y, x] == BLACK:
                positions.append(y)
                # 元画像(検証用)に検出点を赤で色付け
                cv.line(
                    tmp_img,
                    pt1=(x, y),
                    pt2=(x, y),
                    color=(0, 0, 255),
                    thickness=3,
                )
                break
    c = collections.Counter(positions)
    # リストから一番多い出現回数の値(座標)を取得
    margin_bottom = c.most_common(1)[0][0]
    # print("検出されたY座標と回数")
    # print(c.most_common(5))
    # imgshow(tmp_img)
    return margin_bottom


def show_rect(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    # imgshow(hsv)
    min = (0, 0, 0)
    max = (255, 255, 225)
    inrange = cv.inRange(hsv, min, max)
    # 目安として探索する範囲緑枠で表示
    tmp_img = cv.cvtColor(inrange, cv.COLOR_GRAY2BGR)
    cv.rectangle(
        tmp_img,
        pt1=(int(tmp_img.shape[1] * 0.2), int(tmp_img.shape[0] * 0.15)),
        pt2=(int(tmp_img.shape[1] * 0.3), int(tmp_img.shape[0] * 0.8)),
        color=(0, 255, 0),
        thickness=3,
    )
    cv.rectangle(
        tmp_img,
        pt1=(int(tmp_img.shape[1] * 0.9), int(tmp_img.shape[0] * 0.15)),
        pt2=(int(tmp_img.shape[1] * 0.95), int(tmp_img.shape[0] * 0.8)),
        color=(0, 255, 0),
        thickness=3,
    )
    cv.rectangle(
        tmp_img,
        pt1=(margin_left, int(tmp_img.shape[0] * 0.71)),
        pt2=(margin_right, int(tmp_img.shape[0] * 0.95)),
        color=(0, 255, 0),
        thickness=3,
    )
    cv.rectangle(
        tmp_img,
        pt1=(margin_left, 0),
        pt2=(margin_right, margin_bottom),
        color=(0, 0, 255),
        thickness=2,
    )
    imgshow(tmp_img)


# getleft(img)
margin_left = getleft(img)
margin_right = getright(img)
margin_bottom = getbelow(img, margin_left, margin_right)
print(margin_left, " ", margin_right, " ", margin_bottom)
# show_rect(img)

tmp_img = images[0].copy()

# スキル1行分の高さ
skill_height = int(tmp_img.shape[0] * 0.18)

# ↑の赤枠の幅を95%程度にすると丁度良くなる
skill_left = margin_left + int((margin_right - margin_left) * 0.02)
skill_right = margin_left + int((margin_right - margin_left) * 0.98)
cv.rectangle(
    tmp_img,
    pt1=(skill_left, margin_bottom - skill_height),
    pt2=(skill_right, margin_bottom),
    color=(0, 0, 255),
    thickness=2,
)
# imgshow(tmp_img)

img = images[0].copy()
skill_img = img[
    margin_bottom - skill_height : margin_bottom, skill_left:skill_right
]
# imgshow(skill_img)

# 次画像
img = images[1]

# グレースケール化
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
skill_gray = cv.cvtColor(skill_img, cv.COLOR_BGR2GRAY)

print("グレースケール画像")
# imgshow([img_gray, skill_gray])

res = cv.matchTemplate(img_gray, skill_gray, cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
print("\nテンプレートマッチ結果")
print(min_val, max_val, min_loc, max_loc)

# 類似度が最も高い場所を赤枠で囲ってみる
tmp_img = img.copy()
cv.rectangle(
    tmp_img,
    pt1=(max_loc[0], max_loc[1]),
    pt2=(max_loc[0] + skill_img.shape[1], max_loc[1] + skill_img.shape[0]),
    color=(0, 0, 255),
    thickness=2,
)
print("\n類似度が最も高い場所")
# imgshow(tmp_img)

# 結合用に切り抜いてみる
clip_img = img[max_loc[1] + skill_gray.shape[0] : margin_bottom, :]
print("\n結合用に切り抜き")
# imgshow(clip_img)

clip_imgs = []
# 1枚目
clip_imgs.append(images[0][:margin_bottom, :])
# 2枚目
clip_imgs.append(clip_img)

# 2枚目のスキル画像
skill_img = images[1][
    margin_bottom - skill_height : margin_bottom, margin_left:skill_right
]

# 3枚目
img = images[2]

# グレースケール化
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
skill_gray = cv.cvtColor(skill_img, cv.COLOR_BGR2GRAY)

print("グレースケール画像")
# imgshow([img_gray, skill_gray])

res = cv.matchTemplate(img_gray, skill_gray, cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
print("\nテンプレートマッチ結果")
print(min_val, max_val, min_loc, max_loc)

# 類似度が最も高い場所を赤枠で囲ってみる
tmp_img = img.copy()
cv.rectangle(
    tmp_img,
    pt1=(max_loc[0], max_loc[1]),
    pt2=(max_loc[0] + skill_img.shape[1], max_loc[1] + skill_img.shape[0]),
    color=(0, 0, 255),
    thickness=2,
)
print("\n類似度が最も高い場所")
# imgshow(tmp_img)

# 結合用に切り抜いてみる
clip_img = img[max_loc[1] + skill_gray.shape[0] : margin_bottom, :]
print("\n結合用に切り抜き")
# imgshow(clip_img)

clip_imgs.append(clip_img)
len(clip_imgs)

width = clip_imgs[0].shape[1]
height = sum(m.shape[0] for m in clip_imgs)
output_img = np.zeros((height, width, 3), dtype=np.uint8)
y = 0
for clip_img in clip_imgs:
    output_img[y : y + clip_img.shape[0], 0 : clip_img.shape[1]] = clip_img
    y += clip_img.shape[0]
print("結合結果")
# imgshow(output_img)

print(margin_left, margin_right)

clip_img = output_img[:, margin_left:margin_right]
# imgshow(clip_img)
_ = cv.imwrite("vertical.png", clip_img)
