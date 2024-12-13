import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Cargar datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preparar datos
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Arquitectura de red neuronal convolucional
capa_convolucion = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
capa_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
capa_flatten = tf.keras.layers.Flatten()
capa_oculta = tf.keras.layers.Dense(units=64, activation='relu')
capa_salida = tf.keras.layers.Dense(units=10, activation='softmax')

modelo = tf.keras.models.Sequential([capa_convolucion, capa_pooling, capa_flatten, capa_oculta, capa_salida])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(x_train, y_train, epochs=5, batch_size=128)

# Evaluar el modelo
perdida = modelo.evaluate(x_test, y_test)
print('Funcion Perdida:', perdida[0])
print('Precision:', perdida[1])

# Predicción
imagen = cv.imread('./Datos/7.jpg')

# Preprocesamiento de la imagen
# 1. Filtro desenfoque
imagen = cv.GaussianBlur(imagen, (3, 3), 0)

# 2. Escala de grises
imagen = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

# 3. Redimensionar el tamaño a (28, 28)
imagen = cv.resize(imagen, (28, 28))

# 4. Inversión de colores
imagen = cv.bitwise_not(imagen)

# 5. Normalizar la imagen
imagen = imagen.astype('float32') / 255

# 6. Asegurarse de que la imagen tenga la forma correcta para la red neuronal
imagen = np.expand_dims(imagen, axis=-1)  

# 7. Añadir una dimensión de batch (1, 28, 28, 1)
imagen = np.expand_dims(imagen, axis=0)

# Mostrar la imagen preprocesada
plt.imshow(imagen.reshape(28, 28), cmap='gray')
plt.show()

# Realizar la predicción
prediccion = modelo.predict(imagen)
clase = np.argmax(prediccion[0])

print('El número de predicción es:', prediccion)
print('El numero es:', clase)
