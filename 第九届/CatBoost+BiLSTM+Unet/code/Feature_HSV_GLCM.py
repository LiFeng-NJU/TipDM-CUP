# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:44:08 2021

@author: Administrator
"""
import cv2
import numpy as np
import matplotlib.image as mpimg
import os
import scipy.io as sio
from skimage.feature import greycomatrix, greycoprops


# HSV即色调(Hue)、饱和度(Saturation)、亮度(Value)
def get_HSV_GLCM(filename):
    values_temp = []  # 初始化特征

    img = mpimg.imread(filename)

    if img is None:
        return

    input = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2]
    piexs_sum = r * c  # 整个灰度图的像素个数为r*c

    # 遍历灰度图的所有像素
    dark_points = (gray_img < 40)  # 人为设置的超参数,表示0~39的灰度值为暗；这个参数地调到一个适合的区间
    target_array = gray_img[dark_points]
    dark_sum = target_array.size
    dark_prop = dark_sum / (piexs_sum)
    values_temp.append(dark_prop)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB空间转换为HSV空间
    h, s, v = cv2.split(hsv)
    # 一阶矩（均值 mean）
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    values_temp.extend([h_mean, s_mean, v_mean])  # 一阶矩放入特征数组
    # 二阶矩 （标准差 std）
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    values_temp.extend([h_std, s_std, v_std])  # 二阶矩放入特征数组
    # 三阶矩 （斜度 skewness）
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    values_temp.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])  # 三阶矩放入特征数组

    # GLCM
    glcm = greycomatrix(input, [2, 8, 16], [0], 256, symmetric=True, normed=True)
    index = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in index:
        temp = greycoprops(glcm, prop)
        for i in range(3):
            values_temp.append(temp[i][0])

    print(values_temp)
    return values_temp

def file_name(file_dir):
    imageNameList = None
    for root, dirs, files in os.walk(file_dir):
        imageNameList = files
    return imageNameList


file_dir0 = "F:\\TeddyCup\code\\1 DataPreprocessing\\Black_Cut\\"
file_dir1 = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGraySiltyMudStone_Cut\\"
file_dir2 = "F:\\TeddyCup\code\\1 DataPreprocessing\\DarkGreyMudstone_Cut\\"
file_dir3 = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayBlack_Cut\\"
file_dir4 = "F:\\TeddyCup\code\\1 DataPreprocessing\\GrayFineSandstone_Cut\\"
file_dir5 = "F:\\TeddyCup\code\\1 DataPreprocessing\\GraySilestone_Cut\\"
file_dir6 = "F:\\TeddyCup\code\\1 DataPreprocessing\\LightGray_Cut\\"
save_path = "F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data_Cut\\"


if __name__ == '__main__':
    # '''0 Black 黑色煤'''
    feature_Black = []
    path_list = os.listdir(file_dir0)
    path_list.sort(key=lambda x: int(x[:-4]))
    print(path_list)
    N = len(path_list)

    for i in range(0, 9):
        print(i)
        dir_path = file_dir0 + path_list[i]
        temp_ = get_HSV_GLCM(dir_path)
        feature_Black.append(temp_)
    # print(feature_Black)
    # sio.savemat(save_path + 'feature_Black.mat',
    #             {'feature_Black': feature_Black})

    '''1 DarkGraySiltyMudStone 深灰色粉砂质泥岩'''
    # feature_DarkGraySiltyMudStone = []
    # imgName1 = file_name(file_dir1)
    # N = len(imgName1)
    # for i in range(0, N):
    #     print(i)
    #     dir_path = file_dir1 + imgName1[i]
    #     temp_ = get_HSV_GLCM(dir_path)
    #     feature_DarkGraySiltyMudStone.append(temp_)
    # sio.savemat(save_path + 'feature_DarkGraySiltyMudStone.mat',
    #             {'feature_DarkGraySiltyMudStone': feature_DarkGraySiltyMudStone})
    #
    # '''2 DarkGreyMudstone 深灰色泥岩'''
    # feature_DarkGreyMudstone_new = []
    # imgName2 = file_name(file_dir2)
    # N = len(imgName2)
    # for i in range(0, N):
    #     print(i)
    #     dir_path = file_dir2 + imgName2[i]
    #     temp_ = get_HSV_GLCM(dir_path)
    #     feature_DarkGreyMudstone_new.append(temp_)
    # sio.savemat(save_path + 'feature_DarkGreyMudstone_new.mat',
    #             {'feature_DarkGreyMudstone_new': feature_DarkGreyMudstone_new})
    #
    # '''3 GrayBlack 灰黑色泥岩'''
    # feature_GrayBlack = []
    # imgName3 = file_name(file_dir3)
    # N = len(imgName3)
    # for i in range(0, N):
    #     print(i)
    #     dir_path = file_dir3 + imgName3[i]
    #     temp_ = get_HSV_GLCM(dir_path)
    #     feature_GrayBlack.append(temp_)
    # sio.savemat(save_path + 'feature_GrayBlack.mat',
    #             {'feature_GrayBlack': feature_GrayBlack})
    #
    # '''4 GrayFineSandstone 灰色细沙岩'''
    # feature_GrayFineSandstone = []
    # imgName4 = file_name(file_dir4)
    # N = len(imgName4)
    # for i in range(0, N):
    #     print(i)
    #     dir_path = file_dir4 + imgName4[i]
    #     temp_ = get_HSV_GLCM(dir_path)
    #     feature_GrayFineSandstone.append(temp_)
    # sio.savemat(save_path + 'feature_GrayFineSandstone.mat',
    #             {'feature_GrayFineSandstone': feature_GrayFineSandstone})
    #
    # '''5 GraySilestone 灰色泥质粉砂岩'''
    # feature_GraySilestone = []
    # imgName5 = file_name(file_dir5)
    # N = len(imgName5)
    # for i in range(0, N):
    #     print(i)
    #     dir_path = file_dir5 + imgName5[i]
    #     temp_ = get_HSV_GLCM(dir_path)
    #     feature_GraySilestone.append(temp_)
    # sio.savemat(save_path + 'feature_GraySilestone.mat',
    #             {'feature_GraySilestone': feature_GraySilestone})
    #
    # '''6 GraySilestone 浅灰色细砂岩'''
    # feature_LightGray = []
    # imgName6 = file_name(file_dir6)
    # N = len(imgName6)
    # for i in range(0, N):
    #     print(i)
    #     dir_path = file_dir6 + imgName6[i]
    #     temp_ = get_HSV_GLCM(dir_path)
    #     feature_LightGray.append(temp_)
    # sio.savemat(save_path + 'feature_LightGray.mat',
    #             {'feature_LightGray': feature_LightGray})
