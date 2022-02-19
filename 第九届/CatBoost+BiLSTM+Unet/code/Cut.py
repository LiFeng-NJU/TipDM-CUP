# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:14:09 2021

@author: Administrator
"""

import os
import cv2

# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\Black"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGraySiltyMudStone"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGreyMudstone"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayBlack"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayFineSandstone"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GraySilestone"
file_dir = "F:\\TeddyCup\\code\\1 DataPreprocessing\\LightGray"
save_path = "F:\\TeddyCup\code\\1 DataPreprocessing\\LightGray_Cut\\"

'''批量对图片进行剪裁'''
path_list = os.listdir(file_dir)
path_list.sort(key=lambda x:int(x[:-6]))
N = len(path_list)
print(N)
n = 1
for k in range(0, N):
    dir_path = file_dir + "\\" + path_list[k]
    img = cv2.imread(dir_path)
    img_size = img.shape
    width = img_size[0]  # 读取图片的宽度
    height = img_size[1]  # 读取图片的高度
    w = int(width / 3)
    h = int(height / 3)
    print(k)
    # 将一张图片截成4张
    suffix = path_list[k][-4:]  # 图片名称后缀
    for i in range(3):
        for j in range(3):
            img_cut = img[i*w:(i+1)*w, j*h:(j+1)*h]
            cv2.imwrite(save_path + str(n) + suffix, img_cut)
            n += 1
    # img1 = img[0:w, 0:h]
    # img2 = img[w:width, 0:h]
    # img3 = img[0:w, h:height]
    # img4 = img[w:width, h:height]
    #
    # # img1 = img[1000:3000, 500:2500]
    # # img2 = img[500: 3500, 250:2750]
    # suffix = imgName[i][-4:]  # 图片名称后缀
    # cv2.imwrite(save_path + imgName[i][:-4] + '-1cut' + suffix, img1)
    # cv2.imwrite(save_path + imgName[i][:-4] + '-2cut' + suffix, img2)
    # cv2.imwrite(save_path + imgName[i][:-4] + '-3cut' + suffix, img3)
    # cv2.imwrite(save_path + imgName[i][:-4] + '-4cut' + suffix, img4)


