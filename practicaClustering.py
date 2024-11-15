import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

#Leer archivo
datos =pd.read_csv("./Datos/wine_cluster.csv")
print(datos.head())
print(datos.describe())

#Ver la correlacion de las caracteristicas
#sns.pairplot(datos)
#plt.show()

#Caracteristicas
x = datos[['Alcohol','Color_Intensity']]

#Normalizar los datos
x = (x-x.mean())/x.std()

#Eliminar valores nulos
x = x.dropna()

#Variables para la grafica de codo
k_valores = range(2,10)
inercia = []
siluetas = []
for k in k_valores:
    modelo = KMeans(k,random_state=123)
    modelo.fit(x)
    inercia.append(modelo.inertia_)
    silueta = silhouette_score(x,modelo.labels_)
    siluetas.append(silueta)

#Grafica de codo
plt.plot(k_valores,inercia)
plt.title('Grafica de codo')
plt.show()
#Grafica de siluetas
plt.plot(k_valores,siluetas)
plt.title('Grafica de siluetas')
plt.show()

#Crear el modelo (vecinos cercanos)
modelo = KMeans(5,random_state=123)

#Entrenar el modelo
modelo.fit(x)

#Etiquetas y centroides
etiquetas = modelo.labels_
centroides = modelo.cluster_centers_
print(etiquetas)
print(centroides)
print(modelo.inertia_)
silueta = silhouette_score(x,modelo.labels_)
print(silueta)


#Grafico de dispersion
plt.scatter(x['Alcohol'],x['Color_Intensity'],cmap='Blues',label=etiquetas,c=etiquetas)
plt.scatter(centroides[:,0],centroides[:,1],color='red',marker='x',label='centroides')
plt.show()

#Nueva Prediccion
X=np.array([[14,5]])
X=(X-X.mean())/X.std()
prediccion = modelo.predict(X)
print(prediccion)