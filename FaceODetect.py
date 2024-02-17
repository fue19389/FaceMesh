import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# ------------------------------------------------
# ----Extraer data previamente arreglada----------
# ------------------------------------------------

fdat = np.load(r'C:\Users\gerar\PycharmProjects\FACEDATA.npz')

x_train, x_test = fdat['x_train'], fdat['x_test']

# y_test = np.load('y_test.npy')
# y_train = np.load('y_train.npy')
# y_test1 = np.load('y_test1.npy')
# y_train1 = np.load('y_train1.npy')
# y_test2 = np.load('y_test2.npy')
# y_train2 = np.load('y_train2.npy')
y_test3 = np.load('y_test3.npy')
y_train3 = np.load('y_train3.npy')

# 2046 de train
# 744 de test
# ------------------------------------------------

# ------------------------------------------------
# ----Show image of MNIST dataset-----------------
# ------------------------------------------------

# _, axs = plt.subplots(1, 1) / para varias img en una img
#                             / cambiaria las siguientes plt -> axs
# plt.figure(figsize=(28,28)) / esta parte caga el tamaño
# plt.imshow(x_train[0])
# plt.axis('off')
# plt.show()
# plt.pause(10)
# ------------------------------------------------

# --------------------------------------------------
# ----Normalizar datos -----------------------------
# --------------------------------------------------

x_train = x_train / 255
x_test = x_test / 255
# -------------------------------------------------

# ----------------------------------------------------------
# --------Flatten manual-----------------------------------
# ----------------------------------------------------------

# x_train_flattened = x_train.reshape(len(x_train), 28 * 28)
# x_test_flattened = x_test.reshape(len(x_test), 28 * 28)
# ----------------------------------------------------------

# ------------------------------------------------------
# -----Crear modelo con layers -------------------------
# ------------------------------------------------------

n_kernels = 10
layer0 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation='relu', input_shape=(180, 320, 3))
layer1 = tf.keras.layers.MaxPooling2D(2)
layer2 = tf.keras.layers.Flatten()
# layer3 = tf.keras.layers.Dropout(0.5)
layer4 = tf.keras.layers.Dense(75, activation='relu')
layer5 = tf.keras.layers.Dense(150, activation='relu')
layer6 = tf.keras.layers.Dense(75, activation='relu')
# layern = tf.keras.layers.Dense(100, activation='relu')
layer7 = tf.keras.layers.Dense(6, activation='softmax')
model = tf.keras.Sequential([layer0, layer1, layer2, layer4, layer5, layer7])
# ------------------------------------------------------


# ------------------------------------------------------
# ----Compilar, entrenar, evaluar modelo----------------
# ------------------------------------------------------

# optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
# # SET LABELS 0
# history = model.fit(x_train, y_train, epochs=7)
# model.save(r'C:\Users\gerar\PycharmProjects\head_or.keras')
#
# _, actual_acc = model.evaluate(x_test, y_test)

# # SET LABELS 1
# history = model.fit(x_train, y_train1, epochs=7)
# model.save(r'C:\Users\gerar\PycharmProjects\head_or1.keras')
#
# _, actual_acc = model.evaluate(x_test, y_test1)

# # SET LABELS 2
# history = model.fit(x_train, y_train2, epochs=7)
# model.save(r'C:\Users\gerar\PycharmProjects\head_or2.keras')
#
# _, actual_acc = model.evaluate(x_test, y_test2)

# SET LABELS 3
history = model.fit(x_train, y_train3, epochs=7)
model.save(r'C:\Users\gerar\PycharmProjects\head_or3.keras')

_, actual_acc = model.evaluate(x_test, y_test3)

# -----------------------------------------------------

# -----------------------------------------------------
# ----Visualizar el predict del modelo c/u ------------
# -----------------------------------------------------

ndum = int(input('No. de 0 a 743: '))
dummy = x_test[ndum]

plt.imshow(dummy)
plt.axis('off')
plt.show()

y_predicted = model.predict(np.array([dummy]), verbose=1)
print(y_predicted)

prediction = np.argmax(y_predicted)
print(prediction)
# ----------------------------------------------------

# -----------------------------------------------------
# ----Predict del modelo completo ---------------------
# -----------------------------------------------------

y_predicted_full = model.predict(x_test, verbose=2)
prediction_labels = np.zeros_like(y_test3)
for i in range(len(x_test)):
    prediction_labels[i] = np.argmax(y_predicted_full[i])

# ----------------------------------------------------

# -----------------------------------------------------
# ----Matriz de confusión -----------------------------
# -----------------------------------------------------

# # SET LABELS 0
# y_test = np.squeeze(y_test)

# # SET LABELS 1
# y_test1 = np.squeeze(y_test1)

# # SET LABELS 2
# y_test2 = np.squeeze(y_test2)

# SET LABELS 2
y_test3 = np.squeeze(y_test3)

prediction_labels = np.squeeze(prediction_labels)

cm = tf.math.confusion_matrix(labels=y_test3, predictions=prediction_labels)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
# ----------------------------------------------------

