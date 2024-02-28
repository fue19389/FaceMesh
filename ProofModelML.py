# ------------------------------------------------------
# -----------------Librerías a utilizar-----------------
# ------------------------------------------------------

import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# ------------------------------------------------------
# -----------------Selección de modelo------------------
# ------------------------------------------------------

# Actualmente, se tienen modelos del 0 -> 3
n_model = 0

# -----------------------------------------------------

# ------------------------------------------------------
# -------------------Cargar modelo---------------------
# ------------------------------------------------------

if n_model == 0:
    modeldir = r'C:\Users\gerar\PycharmProjects\head_or.keras'
elif n_model == 1:
    modeldir = r'C:\Users\gerar\PycharmProjects\head_or1.keras'
elif n_model == 2:
    modeldir = r'C:\Users\gerar\PycharmProjects\head_or2.keras'
elif n_model == 3:
    modeldir = r'C:\Users\gerar\PycharmProjects\head_or3.keras'
ho_model = tf.keras.models.load_model(modeldir)

# -----------------------------------------------------

# -----------------------------------------------------
# ---------Inicialización de la Webcam---- ------------
# -----------------------------------------------------

# # Capture webcam
#
# cap = cv2.VideoCapture(0)

# Capture webcam & Set resolution

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

# ----------------------------------------------------

# -----------------------------------------------------
# ----Visualizar el predict del modelo LIVE------------
# -----------------------------------------------------

while True:

    # Saving captured image and transforming from BGR TO RGB

    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgRGB = cv2.resize(imgRGB, (320, 180))
    imgRGB = imgRGB.astype(int)/255

    y_predicted = ho_model.predict(np.array([imgRGB]), verbose=None)
    # print(y_predicted)
    prediction = np.argmax(y_predicted)
    print(prediction)

    # Show the complete image
    cv2.imshow('Image', img)

    key = cv2.waitKey(30)
    if key == 27: # 27= Esc
        break