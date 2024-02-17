import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# ------------------------------------------------
# ----Extraer data del MNIST----------------------
# ------------------------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# 60000 de train
# 10000 de test

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

layer0 = tf.keras.layers.Flatten(input_shape=(28, 28))
layer1 = tf.keras.layers.Dense(100, activation='relu')
layer2 = tf.keras.layers.Dense(10, activation='sigmoid')
model = tf.keras.Sequential([layer0, layer1, layer2])
# ------------------------------------------------------


# ------------------------------------------------------
# ----Compilar, entrenar, evaluar modelo----------------
# ------------------------------------------------------

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit(x_train, y_train, epochs=5)

_, actual_acc = model.evaluate(x_test, y_test)
# -----------------------------------------------------

# -----------------------------------------------------
# ----Visualizar el predict del modelo c/u ------------
# -----------------------------------------------------

ndum = int(input('No. de 0 9999: '))
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
prediction_labels = np.zeros_like(y_test)
for i in range(len(x_test)):
    prediction_labels[i] = np.argmax(y_predicted_full[i])

# ----------------------------------------------------

# -----------------------------------------------------
# ----Matriz de confusión -----------------------------
# -----------------------------------------------------

cm = tf.math.confusion_matrix(labels=y_test,
                              predictions=prediction_labels)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
# ----------------------------------------------------
