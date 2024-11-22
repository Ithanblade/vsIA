import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo con datos de entrada
datos = pd.read_excel('Datos/temperatura.xlsx')
print(datos.head())
print(datos.describe())

# sperar características y etiquetas
X = datos[['celsius']]
y = datos['fahrenheit']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

#Crear el modelo
#Perceptron
#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo= tf.keras.Sequential([capa])

#Modelo con capa de entrada y salida
#capa_entrada = tf.keras.layers.Dense(units=1, input_shape=[1])
#capa_salida = tf.keras.layers.Dense(units=1)
#modelo = tf.keras.Sequential([capa_entrada, capa_salida])

#Modelo con capa de entrada, oculta y salida
capa_entrada = tf.keras.layers.Dense(units=1, input_shape=[1])
capa_oculta = tf.keras.layers.Dense(units=3)
capa_salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa_entrada, capa_oculta, capa_salida])


#Compilar el modelo (preparar los datos antes de entrenar)
modelo.compile(optimizer='adam', loss='mean_squared_error')

#Entrenar el modelo
historial=modelo.fit(X_train, y_train, epochs=1000)

#Prediccion
resultado = modelo.predict([np.array([25])])
print(resultado)

#funcion de perdida
perdida = modelo.evaluate(X_test, y_test)
plt.plot(historial.history['loss'])
plt.show()
print(perdida)