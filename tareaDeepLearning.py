import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#Cargar datos
from tensorflow import keras 
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

#Preparar datos
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#Arquitectura de red neuronal convolucional
capa_convolucion = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
capa_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
capa_flatten = tf.keras.layers.Flatten()
capa_oculta = tf.keras.layers.Dense(units=64, activation='relu')
capa_salida = tf.keras.layers.Dense(units=10, activation='softmax')

modelo = tf.keras.models.Sequential([capa_convolucion, capa_pooling, capa_flatten, capa_oculta, capa_salida])

#Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Entrenar el modelo
modelo.fit(x_train, y_train, epochs=8, batch_size=128)

#Evaluar el modelo
perdida = modelo.evaluate(x_test, y_test)
print('Funcion Perdida:', perdida[0])
print('Precision:', perdida[1])

clases_fashion = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

imagen = cv.imread('./Datos/pantalon.jpg', cv.IMREAD_GRAYSCALE)  # Cargar en escala de grises
etiqueta_real = "pantalon"  # La etiqueta real de la imagen

# Preprocesar la imagen
imagen = cv.resize(imagen, (28, 28))  # Redimensionar a 28x28
imagen = imagen.astype('float32') / 255.0  # Normalizar entre 0 y 1
imagen = np.expand_dims(imagen, axis=-1)  # Añadir dimensión del canal (28, 28, 1)
imagen_batch = np.expand_dims(imagen, axis=0)  # Añadir dimensión de batch (1, 28, 28, 1)

# Mostrar la imagen preprocesada
plt.imshow(imagen.squeeze(), cmap='gray')  # `squeeze` elimina dimensiones innecesarias
plt.title(f"Etiqueta Real: {etiqueta_real}")
plt.axis('off')
plt.show()

# Realizar la predicción
prediccion = modelo.predict(imagen_batch)
clase_predicha = np.argmax(prediccion[0])

# Mostrar resultados
print('Predicción del modelo:', clases_fashion[clase_predicha])
print('Etiqueta Real:', etiqueta_real)