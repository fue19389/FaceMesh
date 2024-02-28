# ------------------------------------------------
# ----- Librerías a utilizar ---------------------
# ------------------------------------------------

import os
import cv2
import numpy as np
import pandas as pd

# ------------------------------------------------
# ----- Seleccionar de datos ---------------------
# ------------------------------------------------

# 0 extrae imágenes, 1 extrae etiquetas
imglbl = 0

# 0 extrae grupo de entrenamiento, 1 extrae grupo de prueba
tsttrn = 0

# Si se escoge etiquetas, guardar para modelos 0 -> 3
lbl = 2

# ------------------------------------------------
# ----- Rutina extracción pixeles-----------------
# ------------------------------------------------

if imglbl == 0:

    # Se seleccionan los directorios con las fotografías, por tamaño están fuera del repositorio
    # Las variables exportadas de imágenes se guardan fuera del repositorio por su tamaño
    if tsttrn == 0:
        directory = r"C:\Users\gerar\PycharmProjects\TRAINFACE"
        xdir = r'C:\Users\gerar\PycharmProjects\x_train'
    if tsttrn == 1:
        directory = r"C:\Users\gerar\PycharmProjects\TESTFACE"
        xdir = r'C:\Users\gerar\PycharmProjects\x_test'

    xlist = os.listdir(directory)
    lenolist = int(len(np.array(xlist)))
    x_t = np.zeros((lenolist, 180, 320, 3))
    i = 0

    for filename in xlist:
        # if filename.endswith('.jpg'): / Esto solo aplica si hay mas de un tipo de archivos
        im = cv2.imread(os.path.join(directory, filename))
        imr = cv2.resize(im, (320, 180))
        imr = cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
        x_t[i] = imr
        i = i + 1

    x_t = x_t.astype(int)
    np.save(xdir, x_t)


# ------------------------------------------------
# ----- Rutina extracción etiquetas---------------
# ------------------------------------------------
if imglbl == 1:

    # Se selecciona el archivo .xlsx dentro del repositorio
    lbldir = r"C:\Users\gerar\PycharmProjects\FaceMesh\facelabels.xlsx"

    # Las variables exportadas de etiquetas se guardan dentro del repositorio por su tamaño
    if tsttrn == 0:
        if lbl == 0:
            sheet = 'traintags'
            ydir = 'y_train'
        if lbl == 1:
            sheet = 'traintags1'
            ydir = 'y_train1'
        if lbl == 2:
            sheet = 'traintags2'
            ydir = 'y_train2'
        if lbl == 3:
            sheet = 'traintags3'
            ydir = 'y_train3'

    if tsttrn == 1:
        if lbl == 0:
            sheet = 'testtags'
            ydir = 'y_test'
        if lbl == 1:
            sheet = 'testtags1'
            ydir = 'y_test1'
        if lbl == 2:
            sheet = 'testtags2'
            ydir = 'y_test2'
        if lbl == 3:
            sheet = 'testtags3'
            ydir = 'y_test3'

    y_t = np.array(pd.read_excel(lbldir, sheet_name=sheet))
    y_t = y_t.astype(int)
    np.save(ydir, y_t)