# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:10:46 2021

@author: Administrator
"""

import numpy as np
import random
import os
import cv2

'''
添加椒盐噪声
'''
def salt(src,percentage):
    NoiseImg = src
    rows,cols,_ = NoiseImg.shape
    NoiseNum=int(percentage*rows*cols)
    for i in range(NoiseNum):
        randX=np.random.randint(0,rows-1)
        randY=np.random.randint(0,cols-1)
        if random.randint(0,1) <= 0.5:
            NoiseImg[randX,randY]=255
        else:
            NoiseImg[randX,randY]=NoiseImg[randX,randY]
    return NoiseImg

'''获取文件夹下的图片名称'''
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\Black_Cut"
# file_dir = "F:\\TeddyCup\\code\\1 DataPreprocessing\\DarkGraySiltyMudStone_Cut"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGreyMudstone\\"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayBlack_Cut"
file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayFineSandstone_Cut"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\GraySilestone\\"
# file_dir = "F:\\TeddyCup\code\\1 DataPreprocessing\\LightGray\\"

save_path = "F:\\TeddyCup\\code\\1 DataPreprocessing\\GrayFineSandstone_Cut\\"
'''批量对图片添加椒盐噪声'''
path_list = os.listdir(file_dir)
path_list.sort(key=lambda x:int(x[:-4]))
N = len(path_list)

# imgName = file_name(file_dir)
# N = len(imgName)
n = 163
for i in range(0, N):
    dir_path = file_dir + "\\" + path_list[i]
    img = cv2.imread(dir_path)
    img = salt(img, 0.01)
    cv2.imwrite(save_path + str(n) + ".bmp", img)
    n = n + 1
    print(i)