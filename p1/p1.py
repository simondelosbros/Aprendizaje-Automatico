# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:08:01 2018

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt



'''EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS'''

'''
1. Implementar el algoritmo de gradiente descendente.
'''

def gradiente_descendente(w, lr, f_error, f_grad, epsilon, max_iters):
	i=0
	array_error = [f_error(w)]
	while True and i<max_iters:
		w = w - lr * f_grad(w)
		i+=1
		#print(f_error(w))
		array_error.insert(len(array_error),f_error(w))
		
		if np.abs(f_error(w)) < epsilon:
			break
		
	sol=[w,i,array_error]
	return sol
	
'''
2. Considerar la función E(u, v) = (u^3*e^(v−2)−4v^3*e^−u)^2 . Usar gradiente descendente
para encontrar un mínimo de esta función, comenzando desde el punto (u, v) = (1, 1) y
usando una tasa de aprendizaje η = 0,1.
'''

#loquesea=gradiente_descendente([1,1],0.1,f_error,f_grad,10**(-5),100000)

'''
	a)Calcular analíticamente y mostrar la expresión del gradiente de la función E(u, v)
'''

def f_error(w):
	return (w[0]**3 * np.exp(w[1]-2) - 4*w[1]**3 * np.exp(-w[0]))**2

def deriv_u_f_error(w):
	#2*np.sqrt(f_error) no funciona
	return 2*(w[0]**3 * np.exp(w[1]-2) - 4*w[1]**3 * np.exp(-w[0]))*(3*w[0]**2 * np.exp(w[1]-2) + 4*w[1]**3 * np.exp(-w[0]))

def deriv_v_f_error(w):
	#2*np.sqrt(f_error) no funciona
	return 2*(w[0]**3 * np.exp(w[1]-2) - 4*w[1]**3 * np.exp(-w[0]))*(w[0]**3 * np.exp(w[1]-2) - 12*w[1]**2 * np.exp(-w[0]))

def f_grad(w):
	return np.array([deriv_u_f_error(w),deriv_v_f_error(w)],np.float64)


'''
	b) ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de E(u, v)
inferior a 10^−14? (Usar flotantes de 64 bits)
	c) ¿En qué coordenadas (u, v) se alcanzó por primera vez un valor igual o menor a
10^−14 en el apartado anterior?
'''
print(' ')
print('1.b) ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de E(u, v) inferior a 10^−14?')
epsilon=10**(-14)
a=gradiente_descendente([1,1],0.05,f_error,f_grad,epsilon, 1000000)
print('Número de iteraciones (máximo 1000000):')
print(a[1])
wait = input("Pulsa enter para continuar.")

print(' ')
print('1.c) ¿En qué coordenadas (u, v) se alcanzó por primera vez un valor igual o menor a 10^−14 en el apartado anterior.')
print('Valor de (u,v):')
print(a[0])
wait = input("Pulsa enter para continuar.")

'''
3. Considerar ahora la función f(x,y)=(x-2)^2 + 2(y+2)^2 + 2*sin(2πx)*sin(2πy)
'''

def funcion_2(w):
	return ((w[0]-2)**2 + 2*(w[1]+2)**2 + 2*np.sin(np.pi*w[0])*np.sin(np.pi*w[1]))

def deriv_x_f_2(w):
	return (2 * (2*np.pi * np.cos(2*np.pi*w[0]) * np.sin(2*np.pi*w[1]) + w[0]-2))

def deriv_y_f_2(w):
	return (4 * (np.pi * np.sin(2*np.pi*w[0]) * np.cos(2*np.pi*w[1]) + w[1]+2))

def grad_funcion_2(w):
	return np.array([deriv_x_f_2(w), deriv_y_f_2(w)], np.float64)

'''
	a) Usar gradiente descendente para minimizar esta función. Usar como punto inicial
(x0 = 1, y0 = 1), tasa de aprendizaje η = 0,01 y un máximo de 50 iteraciones. Generar
un gráfico de cómo desciende el valor de la función con las iteraciones. Repetir el
experimento pero usando η = 0,1, comentar las diferencias y su dependencia de η.
'''

print(' ')
print('2.a) Generar un gráfico de cómo desciende el valor de la función con las iteraciones. Repetir el experimento pero usando η = 0,1')
b=gradiente_descendente([1,1],0.01,funcion_2,grad_funcion_2,10**(-5),50)

y3a = b[2]
x3a = range(1, len(y3a)+1)
plt.plot(x3a, y3a)
plt.xlabel('Iteraciones')
plt.ylabel('Valor f_error(w)')
plt.title('Learning rate = 0.01')
plt.show()
wait = input("Pulsa enter para continuar.")

c=gradiente_descendente([1,1],0.1,funcion_2,grad_funcion_2,10**(-30),50)

#print(c[1])
y3a = c[2]
x3a = range(1, len(y3a)+1)
plt.plot(x3a, y3a)
plt.xlabel('Iteraciones')
plt.ylabel('Valor f_error(w)')
plt.title('Learning rate = 0.1')
plt.show()
wait = input("Pulsa enter para continuar.")


'''
	b) Obtener el valor mínimo y los valores de las variables (x, y) en donde se alcanzan
cuando el punto de inicio se fija: (2.1, −2.1), (3, −3), (1.5, 1.5), (1, −1). Generar una
tabla con los valores obtenidos
'''

print(' ')
print('2.b) Obtener el valor mínimo y los valores de las variables (x, y) en donde se alcanzancuando el punto de inicio se fija: (2.1, −2.1), (3, −3), (1.5, 1.5), (1, −1).')

b1=gradiente_descendente([2.1,-2.1],0.01,funcion_2,grad_funcion_2,10**(-5),50)
b2=gradiente_descendente([3,-3],0.01,funcion_2,grad_funcion_2,10**(-5),50)
b3=gradiente_descendente([1.5,1.5],0.01,funcion_2,grad_funcion_2,10**(-5),50)
b4=gradiente_descendente([1,-1],0.01,funcion_2,grad_funcion_2,10**(-5),50)
print('w=(2.1,-2.1)')
print(b1[0])
print('w=(3,-3)')
print(b2[0])
print('w=(1.5,1.5)')
print(b3[0])
print('w=(1,-1)')
print(b4[0])
print(' ')
print(' ')
print(' ')
wait = input("Pulsa enter para continuar.")
'''
4. ¿Cuál sería su conclusión sobre la verdadera dificultad de encontrar el mínimo
global de una función arbitraria?
'''
#Luego la pienso jeje





''' EJERCICIO SOBRE REGRESIÓN LINEAL '''
'''
1. Estimar un modelo de regresión lineal a partir de los datos proporcionados de
dichos números (Intensidad promedio, Simetria) usando tanto el algoritmo de la pseudo-
inversa como Gradiente descendente estocástico (SGD). Las etiquetas serán {−1, 1}, una
para cada vector de cada uno de los números. Pintar las soluciones obtenidas junto con los
datos usados en el ajuste. Valorar la bondad del resultado usando E in y E out (para E out cal-
cular las predicciones usando los datos del fichero de test). (usar
Regress_Lin(datos, label) como llamada para la función (opcional)).
'''

def selecciona_necesarios(n1,n2,tipo):
	if tipo=='train':
		todo_X=np.load('datos/X_train.npy')
		todo_y=np.load('datos/y_train.npy')
	else:
		if tipo=='test':
			todo_X=np.load('datos/X_test.npy')
			todo_y=np.load('datos/y_test.npy')
		else:
			print('Tipo valido train o test.')
			return
	
	X=[]
	y=[]
	
	i=0
	for i in range(len(todo_X)):
		if todo_y[i]==n1:
			X.insert(len(X),[1,todo_X[i][0],todo_X[i][1]])
			y.insert(len(y),-1)
		if todo_y[i]==n2:
			X.insert(len(X),[1,todo_X[i][0],todo_X[i][1]])
			y.insert(len(y),1)
	
	solX=np.array(X, np.float64)
	solY=np.array(y, np.float64)
	
	return solX, solY

def aniade_coordenada(X):
	Xtemp=[]
	
	i=0
	for i in range(len(X)):
		Xtemp.insert(len(Xtemp),[1,X[i][0],X[i][1]])
	
	solX=np.array(Xtemp, np.float64)
	
	return solX


def error_matriz(X,y,w): #E_in
	N=len(X)
	producto=np.dot(X,w)-y
	normalizado=np.linalg.norm(producto)
	return 1/N * normalizado * normalizado

def error_out(X,y,w):
	total=len(X)
	fallos=0
	i=0
	for i in range(total):
		producto=np.dot(X[i],w)
		if producto>0 and y[i]==-1:
			fallos+=1
		if producto<0 and y[i]==1:
			fallos+=1
		
	porcentaje_fallo=1.0*fallos/total
	
	return porcentaje_fallo




def pseudo_inversa(X,y):
	pseudoX=np.dot(np.linalg.inv(
			np.dot(np.transpose(X),X)),np.transpose(X))
	return np.dot(pseudoX,y)	



'''
def set_minibatches(X,y,long):
	posicion=0
	longitud=long
	i=0
	minibatches=[]
	total=len(X)/longitud
	while i<total:
		minibatches.insert(i,(X[posicion:posicion+longitud],y[posicion:posicion+longitud]))
		posicion+=longitud
		i+=1
	print(len(minibatches[1][1]))
	return minibatches
'''

def h(x_,w_):
	sol=np.zeros(len(x_),np.float64)
	i=0
	for i in range(len(sol)):
		sol[i]=np.dot(np.transpose(w_),x_[i])
	return sol

def set_minibatches(X,y,longitud):
	array_index=np.random.permutation(len(X))
	
	Xtemp=[]
	ytemp=[]
		
	i=0
	for i in range(longitud):
		Xtemp.insert(len(Xtemp),X[array_index[i]])
		ytemp.insert(len(ytemp),y[array_index[i]])
	
	_X_=np.array(Xtemp,np.float64)
	_y_=np.array(ytemp,np.float64)
	
	return _X_, _y_


def gradiente_descendente_estocastico(X,y,lr,epsilon,longitud,max_iter):	
	_w_ = np.zeros(len(X[1]),np.float64)
	#mejor_w=np.zeros(len(X[1]),np.float64)
	
	_X_, _y_=set_minibatches(X,y,longitud)
	
	i=0
	while True and i < max_iter:
		suma=np.zeros(len(X[1]), np.float64)
		
		suma += np.dot(np.transpose(_X_),( h(_X_,_w_)-_y_ ))
		_w_=_w_-lr*(2.0/longitud)*suma
		
		if np.abs(error_matriz(X,y,_w_)) < epsilon:
			break
		
		_X_, _y_=set_minibatches(X,y,longitud)
		i+=1
		
		'''if np.abs(error_matriz(X,y,_w_)) <
		      np.abs(error_matriz(X,y,mejor_w)):
				  mejor_w=np.array(_w_,np.float64)'''
	
	return _w_
	#return mejor_w	


X,y=selecciona_necesarios(1,5,'train')
X_test,y_test=selecciona_necesarios(1,5,'test')

w=pseudo_inversa(X,y)

max_val = 1.
t = np.arange(0., max_val+0.5, 0.5)
#plt.plot(t, w[0]+w[1]*t, 'r-')
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
plt.xlabel('X')
plt.ylabel('y')
plt.scatter(X[:, 1],X[:, 2], c=y)
plt.title('Ajuste de los datos mediante pseudo inversa')
plt.show()
wait = input("Pulsa enter para continuar.")
ein_pseudo_inversa=error_matriz(X,y,w)
eout_pseudo_inversa=error_out(X_test,y_test,w)



w=gradiente_descendente_estocastico(X,y,0.05,0.01,64,1000)

max_val = 1.
t = np.arange(0., max_val+0.5, 0.5)
#plt.plot(t, w[0]+w[1]*t, 'r-')
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
plt.xlabel('X')
plt.ylabel('y')
plt.scatter(X[:, 1],X[:, 2], c=y)
plt.title('Ajuste de los datos mediante gradiente descendiente estocástico')
plt.show()

wait = input("Pulsa enter para continuar.")

ein_sgd=error_matriz(X,y,w)
eout_sgd=error_out(X_test,y_test,w)


print(' ')
print('Error con el algoritmo de la pseudo-inversa:')
print('Ein:')
print(ein_pseudo_inversa)
print('Eout:')
print(eout_pseudo_inversa)
print(' ')
print('Error con el algoritmo de gradiente descendiente estocástico:')
print('Ein:')
print(ein_sgd)
print('Eout:')
print(eout_sgd)
print(' ')

wait = input("Pulsa enter para continuar.")

'''
2. En este apartado exploramos como se transforman los errores E in y E out cuando au-
mentamos la complejidad del modelo lineal usado. Ahora hacemos uso de la función
simula_unif(N, 2, size) que nos devuelve N coordenadas 2D de puntos uniformemente
muestreados dentro del cuadrado definido por [−size, size] × [−size, size]
'''

def simula_unif(N=2, dims=2, size=(0, 1)):
	m = np.random.uniform(low=size[0], high=size[1], size=(N, dims))
	
	return m


'''
	a)Generar una muestra de entrenamiento de N = 1000 puntos en el cuadrado
X = [−1, 1] × [−1, 1]. Pintar el mapa de puntos 2D.
'''
#Generate data
X = simula_unif(N=1000, dims=2, size=(-1, 1))
#Plot data
plt.title('Muestra de entrenamiento N=1000 puntos.')
plt.scatter(X[:, 0], X[:, 1])
plt.show()
wait = input("Pulsa enter para continuar.")

'''
	b)Consideremos la función f(x1, x2) = sign((x1 − 0,2)² + x2²− 0,6) que usaremos
para asignar una etiqueta a cada punto de la muestra anterior. Introducimos
ruido sobre las etiquetas cambiando aleatoriamente el signo de un 10 % de las
mismas. Pintar el mapa de etiquetas obtenido.
'''

def f_signo(x1,x2):
	return np.sign((x1-0.2)*(x1-0.2) + x2*x2-0.6)

def label_data(x1, x2):
	y = f_signo(x1,x2)
	idx = np.random.choice(range(y.shape[0]),
		size=(int(y.shape[0]*0.1)), replace=True)
	y[idx] *= -1
	return y
#Generate data
X = simula_unif(N=1000, dims=2, size=(-1, 1))
y = label_data(X[:, 0], X[:, 1])
#Plot data
plt.title('Distribución uniforme y resultados con ruido.')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

wait = input("Pulsa enter para continuar.")

'''
	c) Usando como vector de características (1, x1, x2) ajustar un modelo de regresion
lineal al conjunto de datos generado y estimar los pesos w. Estimar el error de
ajuste Ein usando Gradiente Descendente Estocástico (SGD).
'''

X=aniade_coordenada(X)

w=gradiente_descendente_estocastico(X,y,0.1,0.01,64,1000)

plt.scatter(X[:, 1],X[:, 2], c=y)
max_val = 1.
t = np.arange(-0.5, 0.5,0.1)
#plt.plot(t, w[0]+w[1]*t, 'r-')
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
plt.axis([-1, 1, -1, 1])

plt.xlabel('X')
plt.ylabel('y')
plt.title('Modelo de regresión lineal sobre distribución uniforme con SGD')
plt.show()

wait = input("Pulsa enter para continuar.")

ein_3c=error_matriz(X,y,w)
print('Error para la distribución uniforme con SGD:')
print(ein_3c)

wait = input("Pulsa enter para continuar.")

'''
	d) Ejecutar todo el experimento definido por (a)-(c) 1000 veces (generamos 1000
muestras diferentes) y:
	• Calcular el valor medio de los errores Ein de las 1000 muestras.
	• Generar 1000 puntos nuevos por cada iteración y calcular con ellos el valor 
	  de E out en dicha iteración. Calcular el valor medio de E out en todas las
	  iteraciones.
'''

def media_error_uniforme():
	i=0
	suma_ein=0
	suma_eout=0
	for i in range(1000):
		X = simula_unif(N=1000, dims=2, size=(-1, 1))
		y = label_data(X[:, 0], X[:, 1])
		X=aniade_coordenada(X)
		w=gradiente_descendente_estocastico(X,y,0.05,0.01,64,100)
		suma_ein+=error_matriz(X,y,w)
		suma_eout+=error_out(X,y,w)
		print(i)

	media_ein=suma_ein/1000.0
	media_eout=suma_eout/1000.0
	
	return media_ein, media_eout
	

media_ein, media_eout=media_error_uniforme()

print(' ')
print('Valor medio de error tras 1000 ajustes de una distribución uniforme:')
print('Ein:')
print(media_ein)
print('Eout:')
print(media_eout)
print(' ')
print(' ')

wait = input("Pulsa enter para continuar.")

'''
	e) Valore que tan bueno considera que es el ajuste con este modelo lineal a la vista
de los valores medios obtenidos de E in y E out
'''
#pal PDF


def main():	
#	a=0
#	print(a)
	if __name__ == '__main__':
		main()