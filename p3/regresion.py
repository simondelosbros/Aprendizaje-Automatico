#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:30:27 2018

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.utils import check_random_state
from sklearn.preprocessing import PolynomialFeatures


def readData(file_x, file_y):
	x = np.load(file_x)
	y = np.load(file_y)	
	
	return x, y
	
def pairs(data, names):
	d = data.shape[1]
	fig, axes = plt.subplots(nrows=d, ncols=d, sharex='col', sharey='row')
	for i in range(d):
		for j in range(d):
			ax = axes[i,j]
			if i == j:
				ax.text(0.5, 0.5, names[i], transform=ax.transAxes,
				horizontalalignment='center', verticalalignment='center', fontsize=9)
			else:
				ax.plot(data[:,j], data[:,i], '.k', color='b')
	plt.show()
	input("Pulsa enter para continuar.")

# Leemos el conjunto de entrenamiento
X, y = readData('datos/airfoil_self_noise_X.npy', 'datos/airfoil_self_noise_y.npy')
# Visualizamos los datos en parejas
pairs(np.concatenate((np.array([y]).T, X), axis=1), ['Presión', 'Frecuencia', 'Ángulo', 'Cuerda', 'Velocidad', 'Grosor'])

# Escalamos los datos
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Usamos características polinomiales para añadir información
poly = PolynomialFeatures(2)
X = poly.fit_transform(X) 

# Permutamos los datos antes de separar en train y test
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]

# Separamos los datos en train y test
X_train = X[0:2*X.shape[0]//3]
y_train = y[0:2*y.shape[0]//3]
X_test = X[2*X.shape[0]//3:X.shape[0]]
y_test = y[2*y.shape[0]//3:y.shape[0]]

# Aplicamos K-FOLD
k = 8 # Número de subconjuntos a generar
kf = KFold(n_splits=k) # Inicializamos el K-Fold
mejor_alpha = 0
mejor_media = 0
alphas = []
scores = []

for i in range(-7,7):			
	suma = 0	
	a = i*0.001 # En este caso, alpha crece linealmente
	
	for train_index, test_index in kf.split(X_train):
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		
		# Aplicamos Ridge con el parámetro a
		lr = linear_model.Ridge(alpha = a)
		# Ajustamos el modelo a partir de los datos
		lr.fit(X_train_, y_train_)	
		# Sumamos el porcentaje de acierto
		suma += lr.score(X_test_, y_test_)
		
	media = 1.0*suma/k
	
	# Guardamos alpha y la media para la gráfica
	alphas.append(a)
	scores.append(media)
	
	# Si mejora el porcentaje de acierto con el nuevo alpha
	if media > mejor_media:
		mejor_media = media
		mejor_alpha = a

print('Mejor alpha=', mejor_alpha)
		
plt.plot(alphas,scores)
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.title('Porcentaje de acierto respecto de alpha.')
plt.show()
input("Pulsa enter para continuar.")

# Ajustamos Ridge a partir de los datos
lr = linear_model.Ridge(alpha = mejor_alpha)
model = lr.fit(X_train, y_train)

# Calculamos el score con dicho ajuste
# con el conjunto test generado anteriormente
score = lr.score(X_test, y_test)
print()
print('Porcentaje de acierto sobre los datos test: ', score)
print('E_out=', 1-score)
input("Pulsa enter para continuar.")

# Realizamos predicciones del modelo lineal
predictions= lr.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Valores reales junto a sus predicciones.')
plt.show()
input("Pulsa enter para continuar.")