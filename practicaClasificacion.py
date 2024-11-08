from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

datos =pd.read_csv("./Iris.csv")
datos=datos.drop("Id",axis=1)
print(datos.head())
print(datos.describe())

#Separar las caracteristicas de X y las etiquetas
x = datos.iloc[:,:-1]
y = datos[['Species']]

#Dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)

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
tipo_flor = modelo.predict([[4.5,3.2,1.6,0.3]])
print(f"La flor 1 es de tipo: {tipo_flor}")

tipo_flor = modelo.predict([[3,2.2,2.5,0.5]])
print(f"La flor 2 es de tipo: {tipo_flor}\n")

#Importancia de las caracteristicas Solo para arboles de decision
importancia = modelo.feature_importances_
nombres_columnas = datos.columns.to_list()

for i,k in zip(nombres_columnas,importancia):
    print(f"{i}: {k}")

#Grafico de dispersion con el petalo
sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=datos)
plt.show()


