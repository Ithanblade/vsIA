import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

#Leer archivo
datos =pd.read_csv("./Datos/face.csv")
print(datos.head())
print(datos.describe())

#Ver la correlacion de las caracteristicas
#sns.pairplot(datos)
#plt.show()

#Caracteristicas
x = datos[['y','x']]

#Normalizar los datos
x = (x-x.mean())/x.std()

#Eliminar valores nulos
x = x.dropna()

#Variables para la grafica de codo
k_valores = range(2,11)
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
modelo = KMeans(4,random_state=123)

#Entrenar el modelo
modelo.fit(x)

#Etiquetas y centroides
etiquetas = modelo.labels_
centroides = modelo.cluster_centers_
print(etiquetas)
print(centroides)
print("Inercia: ",modelo.inertia_)
siluetaM = silhouette_score(x,modelo.labels_)
print("Silueta: ",siluetaM)


#Grafico de dispersion
grafica = plt.scatter(x['y'],x['x'],cmap='rainbow',label=etiquetas,c=etiquetas)
plt.scatter(centroides[:,0],centroides[:,1],color='black',marker='x',label='centroides')
plt.colorbar(grafica,label='Cluster')
plt.show()

#Nueva Prediccion
#Prediccion 1
X=np.array([[7,18]])
X=(X-X.mean())/X.std()
prediccion1 = modelo.predict(X)
print("Prediccion 1: ",prediccion1)

#Prediccion 2
X=np.array([[10,15]])
X=(X-X.mean())/X.std()
prediccion2 = modelo.predict(X)
print("Prediccion 2: ",prediccion2)

#Prediccion 3
X=np.array([[8.2,13.2]])
X=(X-X.mean())/X.std()
prediccion3 = modelo.predict(X)
print("Prediccion 3: ",prediccion3)