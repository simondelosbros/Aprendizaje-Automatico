#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:44:10 2018

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures


def readData(file_x, file_y):
	x = np.load(file_x)
	y = np.load(file_y)	
	x_ = np.empty(x.shape, np.float64)
	
	for i in range(0,x.shape[0]):
		for j in range(0,x.shape[1]):
			x_[i][j] = np.float64(1.0*x[i][j]/16.0)
	
	return x_, y


X_train, y_train = readData('datos/optdigits_tra_X.npy', 'datos/optdigits_tra_y.npy')

print('Visualización de algunos dígitos:')

plt.figure(figsize=(10, 5))

for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(np.split(X_train[i], 8), cmap='gray')
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
	
plt.show()
input("Pulsa enter para continuar.")

# Buscamos característias inservibles que eliminar
inutil = VarianceThreshold(threshold=(0.001))
X_train = inutil.fit_transform(X_train)

print('Atributos eliminados=',64-len(X_train[0]))
plt.imshow(np.split(inutil.get_support(), 8), cmap='gray')
plt.title('Atributos inservibles de los datos.')
plt.show()
input("Pulsa enter para continuar.")


# Usamos características polinomiales para añadir información
poly = PolynomialFeatures(2)
X_train= poly.fit_transform(X_train) 

# Aplicamos K-FOLD
k = 2 # Número de subconjuntos a generar
kfold = StratifiedKFold(n_splits=k) # Inicializamos el K-Fold
kfold.get_n_splits(X_train,y_train) # Separamos los conjuntos de entrenamiento
mejor_C = 0
mejor_media = 0
ces = []
medias = []

for i in range(-5,5):
	suma = 0	
	c = 10**i # El crecimiento de nuestro c será exponencial

	for train_index, test_index in kfold.split(X_train,y_train):
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]

		# Aplicamos regresión logística con el parámetro c
		lr = LogisticRegression(C=c)
		# Ajustamos el modelo a partir de los datos
		lr.fit(X_train_, y_train_)
		# Calculamos error respecto de test
		predictions = lr.predict(X_test_)
		# Sumamos el porcentaje de acierto
		suma += lr.score(X_test_, y_test_)

	media = 1.0*suma/k

	# Guardamos la c y la media para la gráfica
	ces.append(c)
	medias.append(media)

	# Si mejora el porcentaje de acierto con el nuevo c
	if media > mejor_media:
		mejor_media = media
		mejor_C = c
		
		
print('C escogido=', mejor_C)
plt.plot(ces,medias)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('score')
plt.title('Porcentaje de acierto respecto de c.')
plt.show()
input("Pulsa enter para continuar.")


# Ajustamos el modelo a partir de los datos
lr = LogisticRegression(C=mejor_C)
lr.fit(X_train, y_train)
# Leemos los datos de test y los preprocesamos igual que los de train
X_test, y_test = readData('datos/optdigits_tes_X.npy', 'datos/optdigits_tes_y.npy')	
X_test = inutil.transform(X_test)
X_test = poly.fit_transform(X_test)	
# Calculamos el score con dicho ajuste
predictions = lr.predict(X_test)	
score = lr.score(X_test, y_test)

print()
print('Valor de acierto con el mejor c: ', score)
print('E_out=', 1-score)
input("Pulsa enter para continuar.")


# Matriz de confusión
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True)
plt.ylabel('Valor real')
plt.xlabel('Valor predecido')
plt.title('Matriz de confusión.', size = 15)
plt.show()
