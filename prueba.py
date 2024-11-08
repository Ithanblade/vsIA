from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

datos =pd.read_csv("./Datos/obesidad.csv")
#Convertir el genero a numerico, 0 para hombre y 1 para mujer
datos=datos.replace('Female',1)
datos=datos.replace('Male',0)
print(datos.head())
print(datos.describe())

#Separar las caracteristicas de X y las etiquetas
x = datos.iloc[:,:-1]
y = datos[['Nobeyesdad']]

#Dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=123)

#Crear el modelo
modelo=DecisionTreeClassifier()

#Entrenar el modelo
modelo.fit(x_train,y_train)

#Determinar las salidas de las x de prueba
y_prediccion = modelo.predict(x_test)

#Metricas de evaluacion
#Exactitud
exactitud = accuracy_score(y_test,y_prediccion)
print(f"La exactitud es: {exactitud}")
#Matriz de confusion
matriz = confusion_matrix(y_test,y_prediccion)
print(f"La matriz de confusion es:\n {matriz}")

#Grafica de matriz de confusion(mapa de calor)
import numpy as np
tags = np.unique(y_test)
import seaborn as sns
sns.heatmap(matriz,annot=True,cmap="Blues",xticklabels=tags,yticklabels=tags)
plt.xlabel('Verdaderos')
plt.ylabel('Predicciones')
plt.title('Matriz de confusion')
plt.show()

#Prediccion con datos nuevos
"""
Persona 1:
•	Edad: 35
•	Género: Femenino
•	Altura: 1.65 m
•	Peso: 70 kg
•	Alcohol: A veces
•	Contenido Calórico: Sí
•	Comidas: 3
•	Monitoreo Calorías: No
•	Fuma: Sí
•	Agua: Medio
•	Historia familiar con sobrepeso: Sí
•	Actividad Física: Frecuentemente
•	Tecnología: Medio
•	Alimento entre comidas: Frecuentemente
"""
nivel_obesidad = modelo.predict([[35,1,1.65,70,1,1,3,0,1,2,1,2,2,2]])
print(f"La persona #1 tiene nivel de obesidad: {nivel_obesidad} \n")

"""
Persona 2:
•	Edad: 45
•	Género: Masculino
•	Altura: 1.75 m
•	Peso: 90 kg
•	Alcohol: Frecuentemente
•	Contenido Calórico: No
•	Comidas: 3
•	Monitoreo Calorías: Sí
•	Fuma: No
•	Agua: Poco
•	Historia familiar con sobrepeso: No
•	Actividad Física: A veces
•	Tecnología: Mucho
•	Alimento entre comidas: A veces
"""
nivel_obesidad = modelo.predict([[45,0,1.75,90,2,0,3,1,0,1,0,1,3,1]])
print(f"La persona #2 tiene nivel de obesidad: {nivel_obesidad} \n")

"""
Persona 3:
•	Edad: 55
•	Género: Masculino
•	Altura: 1.60 m
•	Peso: 90 kg
•	Alcohol: Nunca
•	Contenido Calórico: Sí
•	Comidas: 4
•	Monitoreo Calorías: No
•	Fuma: No
•	Agua: Mucho
•	Historia familiar con sobrepeso: Sí
•	Actividad Física: Nunca
•	Tecnología: Medio
•	Alimento entre comidas: Siempre
"""
nivel_obesidad = modelo.predict([[55,0,1.60,90,0,1,4,0,0,3,1,0,2,3]])
print(f"La persona #3 tiene nivel de obesidad: {nivel_obesidad} \n")

"""
Persona 4:
•	Edad: 25
•	Género: Masculino
•	Altura: 1.80 m
•	Peso: 75 kg
•	Alcohol: Siempre
•	Contenido Calórico: No
•	Comidas: 3
•	Monitoreo Calorías: Sí
•	Fuma: Sí
•	Agua: Medio
•	Historia familiar con sobrepeso: No
•	Actividad Física: Siempre
•	Tecnología: Poco
•	Alimento entre comidas: Nunca
"""
nivel_obesidad = modelo.predict([[25,0,1.80,75,3,0,3,1,1,2,0,3,1,0]])
print(f"La persona #4 tiene nivel de obesidad: {nivel_obesidad} \n")

#Importancia de las caracteristicas Solo para arboles de decision
importancia = modelo.feature_importances_
nombres_columnas = datos.columns.to_list()

for i,k in zip(nombres_columnas,importancia):
    print(f"{i}: {k}")

#Grafico de dispersion con el petalo
sns.scatterplot(x='Height',y='Weight',hue='Nobeyesdad',data=datos)
plt.show()


