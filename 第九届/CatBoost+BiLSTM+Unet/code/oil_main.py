import cv2
import time
import keras
import scipy.ndimage
import tkinter as tk
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from skimage import io
from numpy import *
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter.font as tkFont
from tkinter import messagebox
from skimage.transform import resize
from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import img_to_array, load_img


im_width = 1280
im_height = 720
border = 5

# unet
class TTA_ModelWrapper():
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        pred = []
        for x_i in X:
            p0 = self.model.predict(self._expand(x_i[:, :, 0]))
            p1 = self.model.predict(self._expand(np.fliplr(x_i[:, :, 0])))
            p2 = self.model.predict(self._expand(np.flipud(x_i[:, :, 0])))
            p3 = self.model.predict(self._expand(np.fliplr(np.flipud(x_i[:, :, 0]))))
            p = (p0 +
                 self._expand(np.fliplr(p1[0][:, :, 0])) +
                 self._expand(np.flipud(p2[0][:, :, 0])) +
                 self._expand(np.fliplr(np.flipud(p3[0][:, :, 0])))
                 ) / 4
            pred.append(p)
        return np.array(pred)

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive pat  h
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
def load_model_unet():
    global tta_model
    path_2 = askopenfilename()
    path_model.set(path_2)
    filename = getPath_model.get()
    optimizer = 'adam'
    loss      = bce_dice_loss
    metrics   = [dice_coef]
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    model.load_weights(filename)
    tta_model = TTA_ModelWrapper(model)
    return tta_model
def get_data(path):
    ids = []
    ids.append(path[-9:])
    X = np.zeros((len('1'), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    img = load_img(path, color_mode="rgb")
    x_img = img_to_array(img)
    x_img = resize(x_img, (720, 1280, 1), mode='constant', preserve_range=True)
    X[0, ..., 0] = x_img.squeeze() / 255
    return X

# tkinter
def get_window_size(win, update=True):
    """ 获得窗体的尺寸 """
    if update:
        win.update()
    return win.winfo_width(), win.winfo_height(), win.winfo_x(), win.winfo_y()
def center_window(win, width=None, height=None):
    """ 将窗口屏幕居中 """
    screenwidth = win.winfo_screenwidth()
    screenheight = win.winfo_screenheight()
    if width is None:
        width, height = get_window_size(win)[:2]
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 3)
    win.geometry(size)
def show_confirm(message=""):
    """
        True  : yes
        False : no
    """
    return messagebox.askyesno("确认框", message)
def close(*arg):
    # 退出tkinter
    if show_confirm("确认退出吗 ?"):
        root.destroy()
def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb
def move(event):
    # 窗口移动
    global x,y
    new_x = (event.x-x)+root.winfo_x()
    new_y = (event.y-y)+root.winfo_y()
    s = "1000x800+" + str(new_x)+"+" + str(new_y)
    root.geometry(s)
def button_1(event):
    global x,y
    x,y = event.x,event.y
def my_resize(w, h, w_box, h_box, pil_image):
    # 对图片进行缩放
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def choosepic():
    # 显示岩石无背景二值图像、荧光二值图像、石油所占百分比
    global tta_model
    path_1 = askopenfilename()
    path.set(path_1)
    filename = getPath.get()
    if (filename[-4:]) == ('.bmp'):
        img_open = Image.open(filename)  # 打开getPath获取到的路径的图片
        w, h = img_open.size
        image_resized = my_resize(w, h, w_box, h_box, img_open)
        img_original = ImageTk.PhotoImage(image_resized)
        label.config(image=img_original, width=485)
        label.image = img_original

        # 二值处理
        start = time.time()
        img = io.imread(filename)
        # img = cv2.cvtColor(np.asarray(img_open), cv2.COLOR_RGB2BGR)
        img = img * 1.0  # 将unit8类型转换为float型
        g = img[:, :, 1]  # 提取G通道分量图像
        b = img[:, :, 2]  # 提取B通道分量图像
        result = g - b  # G通道减去B通道
        w = np.multiply(cv2.getGaussianKernel(9, 7), cv2.getGaussianKernel(9, 7).T)
        fg_filter = scipy.ndimage.convolve(result, w, mode='nearest')

        # 二值化
        ret, fg_filter_II = cv2.threshold(fg_filter, 35.0, 255, cv2.THRESH_BINARY)
        dst_open = Image.fromarray(uint8(fg_filter_II))

        w, h = dst_open.size
        image_II = my_resize(w, h, w_box, h_box, dst_open)
        img_II = ImageTk.PhotoImage(image_II)
        label_II.config(image=img_II, width=485)
        label_II.image = img_II

        # 获取灰度图矩阵的行数和列数
        gray_img = fg_filter_II
        r, c = gray_img.shape[:2]
        piexs_sum = r * c  # 整个灰度图的像素个数为r*c
        light_points = (gray_img > 0)  # 获取像素值不为0的像素点的个数
        target_array = gray_img[light_points]
        light_sum = target_array.size
        oilPercentage = (light_sum / (piexs_sum))
        oilPercentage = oilPercentage * 100
        end = time.time()
        second = str(round((end - start), 4))
        oilPre = "石油所占百分比为：" + str(round(oilPercentage, 4)) + "%"
        label_oil.config(text=oilPre, width=30)
        label_oil.text = oilPre
        fpt = "程序所用时间为：" + second +'s'
        label_time.config(text=fpt, width=30)
        label_time.text = fpt

    else:
        # 利用unet模型获取jpg图像的岩石像素点
        start = time.time()
        X_test = get_data(filename)
        preds_test_tta = tta_model.predict(X_test)
        print("succeeful")
        # Threshold predictions
        preds_test_tta_binary = (preds_test_tta > 0.1).astype(np.uint8)
        plt.imshow(preds_test_tta_binary[0].squeeze(), interpolation='bilinear', cmap='gray')
        plt.axis('off')
        plt.savefig("rock_binary_preds.png", bbox_inches='tight', dpi=300, edgecolor="black", pad_inches=0.0)
        img_open = Image.open("rock_binary_preds.png")
        output = img_open.resize((2448, 2048), Image.ANTIALIAS)
        gray_img_jpg = cv2.cvtColor(np.asarray(output), cv2.COLOR_BGR2GRAY)
        light_points_jpg = (gray_img_jpg > 0)  # 获取岩石像素值不为0的像素点的个数
        target_array_jpg = gray_img_jpg[light_points_jpg]
        light_sum_jpg = target_array_jpg.size
        cv2.imwrite('rock_binary_preds_resize.png', gray_img_jpg)

        # 利用通道相减处理图像获取荧光部分的岩石像素点
        img_open = Image.open(filename)  # 打开getPath获取到的路径的图片
        w, h = img_open.size
        image_resized = my_resize(w, h, w_box, h_box, img_open)
        img_original = ImageTk.PhotoImage(image_resized)
        label.config(image=img_original, width=485)
        label.image = img_original

        img = io.imread(filename)
        img = img * 1.0   # 将unit8类型转换为float型
        g = img[:, :, 1]  # 提取G通道分量图像
        b = img[:, :, 2]  # 提取B通道分量图像

        result = g - b    # G通道减去B通道
        # 高斯滤波
        w = np.multiply(cv2.getGaussianKernel(9, 7), cv2.getGaussianKernel(9, 7).T)
        fg_filter = scipy.ndimage.convolve(result, w, mode='nearest')
        ret, fg_filter_II = cv2.threshold(fg_filter, 35.0, 255, cv2.THRESH_BINARY)

        dst_open = Image.fromarray(uint8(fg_filter_II))
        w, h = dst_open.size
        image_II = my_resize(w, h, w_box, h_box, dst_open)
        img_II = ImageTk.PhotoImage(image_II)
        label_II.config(image=img_II, width=485)
        label_II.image = img_II

        # 获取灰度图矩阵的行数和列数
        light_points = (fg_filter_II > 0)  # 获取像素值不为0的像素点的个数
        target_array = fg_filter_II[light_points]
        light_sum = target_array.size
        # light_sum是荧光部分岩石的总数，light_sum_jpg是所有岩石的总数
        oilPercentage = (light_sum / (light_sum_jpg))
        oilPercentage = oilPercentage * 100
        end = time.time()
        oilPre = "石油所占百分比为：" + str(round(oilPercentage, 4)) + "%"
        label_oil.config(text=oilPre, width=30)
        label_oil.text = oilPre

        second = str(round((end - start), 4))
        fpt = "程序所用时间为：" + second + 's'
        label_time.config(text=fpt, width=30)
        label_time.text = fpt

    if img is None:
        return

root = Tk()
root.geometry('1000x820')
center_window(root)
root.title('岩石样本智能识别')

w_box = 485
h_box = 485
path = StringVar()
path_model = StringVar()
global tta_model
# ---------------------------------------------------------------------
# 标题栏设置
canvas = Canvas(root, width=720, height=420)
canvas.configure(bg="lightskyblue")
canvas.pack(expand=YES, fill=BOTH)
f1 = Frame(canvas)
ft0 = tkFont.Font(family="微软雅黑", size=20, weight=tkFont.BOLD)

ft1 = tkFont.Font(family="微软雅黑", size=24, weight=tkFont.BOLD)
canvas_label = tk.Label(f1, text="计算岩石的含油面积百分含量", height=2, fg="white", font=ft1, bg=_from_rgb((39, 38, 54)))
canvas_label.bind("<B1-Motion>", move)
canvas_label.bind("<Button-1>",button_1)
canvas_label.pack(side=tk.LEFT, anchor=NW, expand=tk.YES, fill=tk.X)

im1 = tk.Label(f1, text="Exit", fg="white", font=ft0)
im1.configure(bg=_from_rgb((39, 38, 54)))
im1.bind('<Button-1>', close)
im1.pack(side=tk.RIGHT, fill=tk.Y)
f1.pack(fill=tk.X)

# ---------------------------------------------------------------------
# 主体设置
frm1 = Frame(root)
frm2 = Frame(root)
frm20 = Frame(frm2)
frm21 = Frame(frm2)
frm22 = Frame(frm2)
frm3 = Frame(root)

frm1.config(height=220, width=1000, bg=_from_rgb((39, 38, 54)))
frm1.place(x=0, y=670)
frm1.bind("<B1-Motion>", move)
frm1.bind("<Button-1>",button_1)
ft2 = tkFont.Font(family="微软雅黑", size=14, weight=tkFont.BOLD)
Label(frm1, text='图片路径: ', fg="white", font=ft2, bg=_from_rgb((39, 38, 54))).place(x=30, y=20)
Button(frm1, text='选择图片', command=choosepic, font=ft2, bg="Teal").place(x=810, y=13)  # 按钮：路径选择
getPath = Entry(frm1, state='readonly', text=path, width=70)  # 输入框：路径输入框
getPath.place(x=150, y=27)

Label(frm1, text='模型路径: ', fg="white", font=ft2, bg=_from_rgb((39, 38, 54))).place(x=30, y=80)
Button(frm1, text='选择模型', command=load_model_unet, font=ft2, bg="Teal").place(x=810, y=73)  # 按钮：路径选择
getPath_model = Entry(frm1, state='readonly', text=path_model, width=70)  # 输入框：路径输入框
getPath_model.place(x=150, y=87)

frm2.config(height=500, width=980, bg="lightskyblue")
frm2.place(x=10, y=120)
frm2.bind("<B1-Motion>", move)
frm2.bind("<Button-1>",button_1)

frm21.config(height=485, width=485,  bg="lightskyblue")
frm21.place(x=0, y=0)
Label(frm21, text='原始图像', font=ft2, bg="lightskyblue").place(in_=frm21, anchor=NW)
frm21.bind("<B1-Motion>", move)
frm21.bind("<Button-1>",button_1)
label = Label(frm21, width=w_box, height=h_box, bg="lightskyblue")
label.place(x=0, y=40)

frm20.config(width=2, height=470, bg="black")
frm20.place(x=490, y=0)

frm22.config(height=485, width=485, bg="lightskyblue")
frm22.place(x=495, y=0)
Label(frm22, text='二值图像', font=ft2, bg="lightskyblue").place(in_=frm22, anchor=NW)
frm22.bind("<B1-Motion>", move)
frm22.bind("<Button-1>",button_1)
label_II = Label(frm22, width=w_box, height=h_box, bg="lightskyblue")
label_II.place(x=0, y=40)

frm3.config(height=65, width=980, bg="lightskyblue")
frm3.place(x=10, y=595)
frm3.bind("<B1-Motion>", move)
frm3.bind("<Button-1>",button_1)
label_oil = Label(frm3, width=30, height=3, font=ft2, bg="lightskyblue")
label_oil.place(x=500, y=0)
label_oil.bind("<B1-Motion>", move)
label_oil.bind("<Button-1>",button_1)

label_time= Label(frm3, width=30, height=3, font=ft2, bg="lightskyblue")
label_time.place(x=0, y=0)
label_time.bind("<B1-Motion>", move)
label_time.bind("<Button-1>",button_1)


root.overrideredirect(True)
root.mainloop()