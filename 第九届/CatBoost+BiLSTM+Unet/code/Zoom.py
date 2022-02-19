# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:36:18 2021

@author: Administrator
"""

import os
import cv2

# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\F_Black\\"

# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGraySiltyMudStone\\"
file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGreyMudstone\\"


# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayBlack\\"

# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayFineSandstone\\"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GraySilestone\\"

# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\LightGray\\"

save_path = "F:\\TeddyCup\code\\1 DataPreprocessing\\F_DarkGreyMudstone_new\\Zoom"
def file_name(file_dir):
    imageNameList = None
    for root, dirs, files in os.walk(file_dir):
        imageNameList = files
    return imageNameList


'''批量对图片进行缩放'''
imgName = file_name(file_dir)
N = len(imgName)
for i in range(0, N):
    dir_path = file_dir + imgName[i]
    img = cv2.imread(dir_path)
    height, width, _ = img.shape

    dstHeight = int(height * 0.5)
    dstWeight = int(width * 0.5)

    img = cv2.resize(img, (dstWeight, dstHeight))
    suffix = imgName[i][-4:]  # 图片名称后缀
    cv2.imwrite(save_path + imgName[i][:-4] + suffix, img)


