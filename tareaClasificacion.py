from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

datos =pd.read_excel("./medicinas.xlsx")
datos=datos.drop("Id",axis=1)
print(datos.head())
print(datos.describe())

#Separar las caracteristicas de X y las etiquetas
x = datos.iloc[:,:-1]
y = datos[['Drug']]

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
#Hombre, 36 años, con presión arterial alta, colesterol normal y concentración de sodio y potasio de 10.
tipo_medicamento = modelo.predict([[36,0,2,0,10]])
print(f"Tipo de medicamento para paciente #1: {tipo_medicamento} \n")

#Mujer, 35 años, con presión arterial normal, colesterol normal y concentración de sodio y potasio de 25.
tipo_medicamento = modelo.predict([[35,1,1,0,25]])
print(f"Tipo de medicamento para paciente #2: {tipo_medicamento} \n")

#Mujer, 75 años, con presión arterial bajo, colesterol alto y concentración de sodio y potasio de 9.
tipo_medicamento = modelo.predict([[75,1,0,1,9]])
print(f"Tipo de medicamento para paciente #3: {tipo_medicamento} \n")

#Importancia de las caracteristicas Solo para arboles de decision
importancia = modelo.feature_importances_
nombres_columnas = datos.columns.to_list()

for i,k in zip(nombres_columnas,importancia):
    print(f"{i}: {k}")

#Grafico de dispersion con el petalo
sns.scatterplot(x='Na_to_K',y='BP',hue='Drug',data=datos)
plt.show()


