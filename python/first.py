import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# 画像読み込み
tuples = [
    ("IMREAD_COLOR", cv.IMREAD_COLOR),
    ("IMREAD_UNCHANGED", cv.IMREAD_UNCHANGED),
    ("IMREAD_GRAYSCALE", cv.IMREAD_GRAYSCALE),
]

for tuple in tuples:
    print(tuple[0])
    img = cv.imread("pochi_256_256.png", tuple[1])
    print(f"type:{type(img)} ndim:{img.ndim} shape:{img.shape}\n")
