from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

datos =pd.read_csv("Salary_Data1.csv")
print(datos.head())
print(datos.describe())

#Separar las caracteristicas de X y las etiquetas
x = datos[["YearsExperience"]]
y = datos[["Salary"]]

#Dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

#Crear el modelo
modelo=LinearRegression()

#Entrenar el modelo
modelo.fit(x_train,y_train)

#Evaluar modelo
y_prediccion = modelo.predict(x_test)

#Metricas de evaluacion
mse = mean_squared_error(y_test,y_prediccion)
r2 = r2_score(y_test,y_prediccion)

print(f"El error cuadratico medio es: {mse}")
print(f"El coeficiente de determinacion es: {r2}")

plt.scatter(x,y,color="blue")
plt.plot(x_test,y_prediccion,color="red")
plt.title("Regresion Lineal")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")

#Predicciones con datos nuevos
prediccion = modelo.predict([[2]])
print("Prediccion:", prediccion)

#B0 y coeficiente
print("Coeficiente:", modelo.coef_)
print("Intercepto:", modelo.intercept_)

salario = "modelo.intercept_ + modelo.coef_ * 2"
print("Ecuacion de Regresion: salario = modelo.intercept_ + modelo.coef_ * 2")



