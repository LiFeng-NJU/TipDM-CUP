图像扩增代码：
Cut.py
Notice.py
Zoom.py


特征提取代码：
Feature_HSV_GLCM.py

生成分类csv格式数据代码：
train_csv.m

Catboost模型训练代码：
Rock_Train_Catboost.py

交互式界面岩石分类代码(适用于Catboost模型和BiLSTM模型)：
BMP+JPG: Rock_main.py
JPG:     Rock_main_jpg.py

交互式界面计算岩石含油量：
oil_main.py


交互式界面代码运行后：
先选择【加载模型】，再【选择图片】
模型只要加载一次就好