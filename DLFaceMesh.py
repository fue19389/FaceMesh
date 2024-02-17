# ============================================================================
# MT3006 - LABORATORIO 4
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar una
# red neuronal convolucional simple, para encontrar un modelo que permita
# clasificar imágenes de las letras A, B y C.
# ============================================================================
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
from scipy import io

# Se ajusta el tamaño de letra y de figura
rcParams.update({'font.size': 18})
pyplot.rcParams['figure.figsize'] = [12, 12]

# Se carga la data de entrenamiento y validación desde los archivos .mat
lettersTrain = io.loadmat('lettersTrainSet.mat')
lettersTest = io.loadmat('lettersTestSet.mat')

# Se extraen las observaciones de entrenamiento y validación. La data importada
# presenta las dimensiones (alto, ancho, canales, batch). En este caso se tiene
# solo un canal dado que las imágenes son en escala de grises.
XTrain = lettersTrain['XTrain']
XTest = lettersTest['XTest']
# Se extraen las labels de entrenamiento y validación, estas están dadas en
# forma de un array de chars indicando la letra a la que corresponden: 'A',
# 'B' y 'C'.
TTrain = lettersTrain['TTrain_cell']
TTest = lettersTest['TTest_cell']

# Se obtiene un vector con 20 índices aleatorios entre 0 y 1500-1 para obtener
# los ejemplos de imagen a visualizar.
perm = np.random.permutation(1500)[:20]

# En esta etapa: re-arregla la data de entrada para que presente las dimensiones (batch,
# alto, ancho, canales), ya que es la forma en la que Keras la espera por
# defecto
XTrain = np.transpose(XTrain, axes=[3, 0, 1, 2])
XTest = np.transpose(XTest, axes=[3, 0, 1, 2])

# Se grafican 20 ejemplos de imagen seleccionados aleatoriamente
fig, axs = pyplot.subplots(4, 5)
axs = axs.reshape(-1)

for j in range(len(axs)):
    axs[j].imshow(np.squeeze(XTrain[perm[j], :, :, :]), cmap='gray')
    axs[j].axis('off')
    pyplot.draw()
    pyplot.pause(0.2)

# Se extraen las categorías como los valores únicos (diferentes) del array
# original de labels
classes = np.unique(TTrain)
# Se crean arrays de ceros con las mismas dimensiones de los arrays originales
# de labels
YTrainLabel = np.zeros_like(TTrain)
YTestLabel = np.zeros_like(TTest)

# Se convierte la categoría desde una letra 'A', 'B', 'C' a un número 0, 1 o 2
# respectivamente
for nc in range(len(classes)):
    YTrainLabel[TTrain == classes[nc]] = nc
    YTestLabel[TTest == classes[nc]] = nc

# Se elimina la dimensión "adicional" de los vectores para poder hacer un
# one-hot encoding con la misma en Keras
YTrainLabel = YTrainLabel.reshape(-1)
YTestLabel = YTestLabel.reshape(-1)

# Se efectúa un one-hot encoding para las labels
YTrain = tf.keras.utils.to_categorical(YTrainLabel)
YTest = tf.keras.utils.to_categorical(YTestLabel)

# COMPLETAR: definición, entrenamiento y evaluación del modelo.
# NOTA: durante la predicción puede emplear la función argmax de numpy para
# deshacer el one-hot encoding

# definir el número de kernels
n_kernels = 16

# definición de capas de la red

layer1 = tf.keras.layers.Conv2D(n_kernels, (10, 10), activation='relu', input_shape=(28, 28, 1))
layer2 = tf.keras.layers.MaxPooling2D((6, 6), strides=(6, 6),)
layer3 = tf.keras.layers.Dropout(0.5)
layer4 = tf.keras.layers.Flatten()
layer5 = tf.keras.layers.Dense(64, activation='relu')
layer6 = tf.keras.layers.Dense(3, activation='softmax')
# model = tf.keras.Sequential([layer1,layer2,layer3,layer5])
model = tf.keras.Sequential([layer1, layer2, layer3, layer4, layer5, layer6])
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# compilación del modelo

model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

# *****************************************************************************
# ENTRENE EL MODELO AQUÍ
# ******************************************************************************

history = model.fit(XTrain, YTrain, validation_data=(XTest, YTest), epochs=30)

# Se evalúa el modelo para encontrar la exactitud de la clasificación (tanto
# durante el entrenamiento y la validación)
_, train_acc = model.evaluate(XTrain, YTrain, verbose=0)
_, val_acc = model.evaluate(XTest, YTest, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, val_acc))

# Se grafica la evolución de la pérdida durante el entrenamiento y la
# validación
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# Se grafica la evolución de la exactitud durante el entrenamiento y la
# validación
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# predicción del modelo

Example = XTrain[1, :, :, :]

fig, axs = pyplot.subplots(1, 1)
axs.imshow(np.squeeze(Example), cmap='gray')
axs.axis('off')
pyplot.draw()
pyplot.pause(10)

prueba = np.reshape(Example, (28, 28))
prediccion = model.predict(np.array([prueba]))

# predic_int = np.round(prediccion).astype(int)
print("la predicción es", prediccion)

# Get the weights from the trained model
pesos = model.get_weights()  # Assuming it's the first element in the list
pesos = np.array(pesos[0])

# Se grafican los 16 kernels
fig, axs = pyplot.subplots(4, 4)
axs = axs.reshape(-1)

for j in range(len(axs)):
    axs[j].imshow(np.squeeze(pesos[:, :, :, j]), cmap='gray')
    axs[j].axis('off')
    pyplot.draw()
    pyplot.pause(1)
