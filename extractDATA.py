from PIL import Image
import os
import cv2
import pickle as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow import keras

i = 0

directory = r"C:\Users\gerar\PycharmProjects\TRAINFACE"
xlist = os.listdir(directory)
lenolist = int(len(np.array(xlist)))

# protoimg = cv2.imread(os.path.join(directory, xlist[i]))
# lenoimg0 = len(protoimg)
# lenoimg1 = len(protoimg[i])
# lenoimg2 = len(protoimg[i][i])

# x_train = np.zeros((lenolist, lenoimg0, lenoimg1, lenoimg2))
x_train = np.zeros((lenolist, 180, 320, 3))

for filename in xlist:
    if filename.endswith('.jpg'):
        im = cv2.imread(os.path.join(directory, filename))
        imr = cv2.resize(im, (320, 180))
        imr = cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
        x_train[i] = imr
        i = i + 1
i = 0

directory = r"C:\Users\gerar\PycharmProjects\TESTFACE"
xlist = os.listdir(directory)
lenolist = int(len(np.array(xlist)))
# x_test = np.zeros((lenolist, lenoimg0, lenoimg1, lenoimg2))
x_test = np.zeros((lenolist, 180, 320, 3))

for filename in xlist:
    if filename.endswith('.jpg'):
        im = cv2.imread(os.path.join(directory, filename))
        imr = cv2.resize(im, (320, 180))
        imr = cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
        x_test[i] = imr
        i = i + 1
i = 0

# y_train = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                  sheet_name='traintags'))
# y_test = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                 sheet_name='testtags'))

# y_train1 = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                  sheet_name='traintags1'))
# y_test1 = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                 sheet_name='testtags1'))

# y_train2 = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                  sheet_name='traintags2'))
# y_test2 = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                 sheet_name='testtags2'))

y_train3 = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
                                 sheet_name='traintags3'))
# y_test3 = np.array(pd.read_excel(r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx",
#                                 sheet_name='testtags3'))

x_test = x_test.astype(int)
x_train = x_train.astype(int)
# y_train = y_train.astype(int)
# y_test = y_test.astype(int)
# y_train1 = y_train1.astype(int)
# y_test1 = y_test1.astype(int)
# y_train2 = y_train2.astype(int)
# y_test2 = y_test2.astype(int)
y_train3 = y_train3.astype(int)
# y_test3 = y_test3.astype(int)

np.savez(r'C:\Users\gerar\PycharmProjects\FACEDATA', x_train=x_train, x_test=x_test)
np.save('y_train3', y_train3)
# np.save('y_test3', y_test3)