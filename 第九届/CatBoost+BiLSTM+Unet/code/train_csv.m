clc
clear
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_Black.mat');
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_DarkGraySiltyMudStone.mat')
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_DarkGreyMudstone_new.mat')
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_GrayBlack.mat')
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_GrayFineSandstone.mat')
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_GraySilestone.mat')
load('F:\\TeddyCup\\code\\2 FeatureExtraction\\Feature_data\\feature_LightGray.mat');

class_0 = zeros(350,1);
class_1 = ones(350,1);

class_2 = ones(350,1);
class_3 = ones(350,1);
class_4 = ones(350,1);
class_5 = ones(350,1);
class_6 = ones(350,1);

class_2(:,1) = 2;
class_3(:,1) = 3;
class_4(:,1) = 4;
class_5(:,1) = 5;
class_6(:,1) = 6;

Black_label = [feature_Black class_0];
DarkGraySiltyMudStone_label = [feature_DarkGraySiltyMudStone class_1];
DarkGreyMudstone_label = [feature_DarkGreyMudstone_new class_2];
GrayBlack_label = [feature_GrayBlack class_3];
GrayFineSandstone_label = [feature_GrayFineSandstone class_4];
GraySilestone_label = [feature_GraySilestone class_5];
LightGray_label = [feature_LightGray class_6];

label1 = cat(1, Black_label, DarkGraySiltyMudStone_label);
label2 = cat(1, label1, DarkGreyMudstone_label);
label3 = cat(1, label2, GrayBlack_label);
label4 = cat(1, label3, GrayFineSandstone_label);
label5 = cat(1, label4, GraySilestone_label);
label = cat(1, label5, LightGray_label);
csvwrite('F:\\TeddyCup\\code\\3 TrainModel\\train_data\\label_HSV_GLCM_Cut.csv',label,0,0);
