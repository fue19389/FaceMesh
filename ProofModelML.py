import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# ------------------------------------------------------
# -------------------Cargar modelo---------------------
# ------------------------------------------------------

# # SET LABELS 0
# ho_model = tf.keras.models.load_model('head_or.keras')
# SET LABELS 1
ho_model = tf.keras.models.load_model('head_or1.keras')

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

    success, img = cap.read()
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
print('hola')