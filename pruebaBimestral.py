import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo con datos de entrada
datos = pd.read_excel('Datos/Kilocalorias.xlsx')
print(datos.head())
print(datos.describe())

# seperar características y etiquetas
x = datos[['Peso', 'Tiempo']] 
y = datos['Kilocalorias']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)

#Modelo con capa de entrada, oculta y salida
capa_entrada = tf.keras.layers.Dense(units=2, input_shape=[2])
capa_ocult1 = tf.keras.layers.Dense(units=13, activation='relu')
capa_oculta2 = tf.keras.layers.Dense(units=13, activation='relu')
capa_salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa_entrada, capa_ocult1,capa_oculta2, capa_salida])


#Compilar el modelo (preparar los datos antes de entrenar)
modelo.compile(optimizer='adam', loss='mean_squared_error')

#Entrenar el modelo
historial=modelo.fit(X_train, y_train, epochs=1500)

#Prediccion 1 Kilocalorias quemadas
resultado_1 = modelo.predict(np.array([[72, 40]]))
print("Para una persona con 72 kg y un tiempo de 40 minutos corriendo a 10.8 km/h, se estima que quemaría aproximadamente" + str(resultado_1) + "kilocalorias")

#Prediccion 2 Kilocalorias quemadas
resultado_2 = modelo.predict(np.array([[62, 15]]))
print("Para una persona con 62 kg y un tiempo de 15 minutos corriendo a 10.8 km/h, se estima que quemaría aproximadamente" + str(resultado_2) + "kilocalorias")

#Prediccion 3 Kilocalorias quemadas
resultado_3 = modelo.predict(np.array([[85, 30]]))
print("Para una persona con 85 kg y un tiempo de 30 minutos corriendo a 10.8 km/h, se estima que quemaría aproximadamente" + str(resultado_3) + "kilocalorias")

#funcion de perdida
perdida = modelo.evaluate(X_test, y_test)
plt.plot(historial.history['loss'])
plt.title('Modelo de perdida')
plt.show()
print(perdida)
