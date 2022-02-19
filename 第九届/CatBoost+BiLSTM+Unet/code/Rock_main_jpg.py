import tkinter as tk
from tkinter import ttk
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
import joblib
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings("ignore")
import tkinter.font as tkFont
from tkinter import messagebox
import cv2
import matplotlib.image as mpimg
from skimage.feature import greycomatrix, greycoprops
from keras.models import load_model

# 特征提取
def color_moments(filename):
    # HSV + GLCM
    values_temp = []  # 初始化颜色特征
    input = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # 读取图像，灰度模式
    img = mpimg.imread(filename)
    if img is None:
        return

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2]
    dark_sum = 0  # 偏暗的像素 初始化为0个
    dark_prop = 0 # 偏暗像素所占比例初始化为0
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

    glcm = greycomatrix(input, [2, 8, 16], [0], 256, symmetric=True, normed=True)
    GLCM_index = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in GLCM_index:
        temp = greycoprops(glcm, prop)
        for i in range(3):
            values_temp.append(temp[i][0])
    values_temp = np.array(values_temp)
    result = values_temp.reshape((1, 28))  # 覆盖原来的数据将新的结果给原来的数组
    return result


# 对准确率进行归一化
def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


# 加载模型
# def loadModelPkl():
#     path_2 = askopenfilename()
#     path_mdoel.set(path_2)

# 加载模型
def loadModelPkl():
    path_2 = askopenfilename()
    path_mdoel.set(path_2)
    global model
    if path_2[-2:] == 'kl':
        data_pkl = open(path_2, "rb")
        xgb_model_loaded = joblib.load(data_pkl)
        model = xgb_model_loaded
        print("Catboost模型加载成功！")
    elif path_2[-2:] == 'h5':
        H5model = load_model(path_2)
        model = H5model
        print("LSTM模型加载成功！")
    else:
        print("模型格式不符合，请重新选择h5或pkl格式模型！")

def loadModel(x_test):
    if str(model)[1:4] == 'ker': # LSTM预测
        x_data = np.reshape(x_test, (1, 28, 1))
        predict = model.predict(x_data)
        acc = list(np.array(predict).flatten())
        normal_acc = np.reshape(acc, (1, 7))
        y_pred = np.argmax(predict)  # 取最大值的位置
        return y_pred, normal_acc
    elif str(model)[1:4] == 'cat': # catboost预测
        y_pred = model.predict(x_test)
        # 以概率形式预测
        y_probability = model.predict_proba(x_test)
        # 归一化
        min_number = np.min(y_probability)
        max_number = np.max(y_probability)
        normal_acc = MaxMinNormalization(y_probability, max_number, min_number)
        return y_pred, normal_acc


# 显示测试准确率
def application():
    rock = ['黑色煤', '深灰色泥岩', '浅灰色细砂岩']
    testPic = path.get()  # 获取图片的路径
    testPicArr = color_moments(testPic)  # 形式化图片符合神经网络的输入格式
    preRock, acc = loadModel(testPicArr)
    tex = rock[int(preRock)]
    label2.config(text=tex)
    label2.text = tex
    acc = list(np.array(acc).flatten())
    preValueout = acc
    for row in tree.get_children():
        tree.delete(row)
    for i in range(3):
        tree.insert('', i, values=(rocks[i], str(round(preValueout[i], 5))))
    tree.grid(padx=0, pady=10)


# 显示图片
def choosepic():
    path_1 = askopenfilename()
    path.set(path_1)
    # print(getPath.get())
    img_open = Image.open(getPath.get())  # 打开getPath获取到的路径的图片
    w, h = img_open.size
    image_resized = resize(w, h, w_box, h_box, img_open)
    img = ImageTk.PhotoImage(image_resized)
    label.config(image=img, width=550)
    label.image = img  # keep a reference


# 对图片进行缩放
def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


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


# 将int的rgb元组转换为tkinter颜色代码
def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb

# 去除窗体的边框
def show_title(*args):
    no_title = True
    root.overrideredirect(no_title)
    no_title = not no_title

def show_confirm(message=""):
    """
		True  : yes
		False : no
	"""
    return messagebox.askyesno("确认框", message)


# 退出tkinter
def close(*arg):
    if show_confirm("确认退出吗 ?"):
        root.destroy()

# 窗口移动
def move(event):
    global x,y
    new_x = (event.x-x)+root.winfo_x()
    new_y = (event.y-y)+root.winfo_y()
    s = "1000x800+" + str(new_x)+"+" + str(new_y)
    root.geometry(s)
def button_1(event):
    global x,y
    x,y = event.x,event.y


# 初始化一个窗口
root = Tk()
root.geometry('1000x820')
center_window(root)
root.title('岩石样本智能识别')

# 初始化所需变量
w_box = 500
h_box = 500
path = StringVar()
path_mdoel = StringVar()
imgGet = StringVar()
global model

# ---------------------------------------------------------------------
# 标题栏设置
canvas = Canvas(root, width=720, height=420)
canvas.configure(bg=_from_rgb((39, 38, 54)))
canvas.pack(expand=YES, fill=BOTH)
f1 = Frame(canvas)
ft0 = tkFont.Font(family="微软雅黑", size=20, weight=tkFont.BOLD)

ft1 = tkFont.Font(family="微软雅黑", size=24, weight=tkFont.BOLD)
canvas_label = tk.Label(f1, text="岩石智能识别演示", height=2, fg="white", font=ft1, bg=_from_rgb((39, 38, 54)))
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
frm21 = Frame(frm2)
frm22 = Frame(frm2)
frm3 = Frame(root)
# 选择图片路径
frm1.config(height=100, width=980, bg=_from_rgb((39, 38, 54)))
frm1.place(x=10, y=620)
ft2 = tkFont.Font(family="微软雅黑", size=14, weight=tkFont.BOLD)
Label(frm1, text='图片路径: ', fg="white", font=ft2, bg=_from_rgb((39, 38, 54))).place(x=30, y=30)
Button(frm1, text='选择图片', command=choosepic, font=ft2, bg="Teal").place(x=810, y=23)  # 按钮：路径选择，调用函数choosepic
getPath = Entry(frm1, state='readonly', text=path, width=70)  # 输入框：路径输入框
getPath.place(x=150, y=37)
frm1.bind("<B1-Motion>", move)
frm1.bind("<Button-1>",button_1)

# 选择pkl模型路径
frm3.config(height=100, width=980, bg=_from_rgb((39, 38, 54)))
frm3.place(x=10, y=700)
ft2 = tkFont.Font(family="微软雅黑", size=14, weight=tkFont.BOLD)
Label(frm3, text='模型路径: ', fg="white", font=ft2, bg=_from_rgb((39, 38, 54))).place(x=30, y=30)
Button(frm3, text='加载模型', command=loadModelPkl, font=ft2, bg="Teal").place(x=810, y=23)  # 按钮：路径选择，调用函数choosepic
getModelPath = Entry(frm3, state='readonly', text=path_mdoel, width=70)  # 输入框：路径输入框
getModelPath.place(x=150, y=37)
frm3.bind("<B1-Motion>", move)
frm3.bind("<Button-1>",button_1)
# 显示图片
frm2.config(height=500, width=980, bg="black")
frm2.place(x=10, y=120)
frm21.config(height=480, width=600, bg=_from_rgb((9, 66, 119)))
frm21.place(x=10, y=10)
label = Label(frm21, width=w_box, height=h_box, bg=_from_rgb((9, 66, 119)))
label.place(x=25, y=0)
label.bind("<B1-Motion>", move)
label.bind("<Button-1>",button_1)

# 显示测试准确率
frm22.config(height=500, width=290, bg=_from_rgb((9, 66, 119)))
frm22.place(x=615, y=10)
frm22.bind("<B1-Motion>", move)
frm22.bind("<Button-1>",button_1)
ft3 = tkFont.Font(family="宋体", size=15, weight=tkFont.BOLD)

tree = ttk.Treeview(frm22, columns=('岩石种类', '准确率'), show='headings')
style = ttk.Style()
style.configure("Treeview.Heading", font=ft3)
style.configure("Treeview", font=("黑体", 13))

tree.column('岩石种类', width=200, anchor='center')
tree.column('准确率', width=150, anchor='center')
tree.heading('岩石种类', text='岩石种类')
tree.heading('准确率', text='准确率')

rocks = ['黑色煤', '深灰色泥岩', '灰色细砂岩']

preValueout = [0, 0, 0]
for i in range(3):
    tree.insert('', i, values=(rocks[i], str(preValueout[i])))
tree.grid(padx=0, pady=10)

Label(frm22, text="网络预测的岩石是:", anchor=NW, fg="white", font=ft3, bg=_from_rgb((9, 66, 119))).place(x=0, y=250)
label2 = Label(frm22, anchor=NW, fg="red", font=ft3, bg=_from_rgb((9, 66, 119)))
label2.grid(padx=0, pady=40)
ft4 = tkFont.Font(family="微软雅黑", size=12, weight=tkFont.BOLD)
Button(frm22, text="测试", command=application, font=ft4, bg="Teal").grid(pady=37)
root.overrideredirect(True)
root.mainloop()
