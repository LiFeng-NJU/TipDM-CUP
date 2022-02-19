# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:24:21 2018
@author: Administrator
"""
from sklearn.model_selection import train_test_split
import catboost as cb
from numpy import loadtxt
import joblib
import numpy as np
import matplotlib.pyplot as plt

dataset = loadtxt('train_data\\label_HSV_GLCM_BMP_JPG.csv', delimiter=",")
X = dataset[:, 0:28]
y = dataset[:, 28]

# Put the order in order
np.random.seed(116)
np.random.shuffle(X)
np.random.seed(116)
np.random.shuffle(y)

model = cb.CatBoostClassifier(iterations=500,
                              random_strength=5,
                              depth=6,
                              learning_rate=0.5)
# jpg:random_state=4, acc=0.9259
# bmp:random_state=3, acc=0.8581
train_x, test_x, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
model.fit(train_x, y_train)
# save model
joblib.dump(model, 'model_weight\\CatBoost_model_BMP_JPG.pkl')

print("accuracy on the training subset:{:.4f}".format(model.score(train_x, y_train)))
print("accuracy on the test subset:{:.4f}".format(model.score(test_x, y_test)))

