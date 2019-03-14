#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:44:10 2018

@author: simon
"""

#FUNCIONES NECESARIAS
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(13)

def simula_unif(N=2, dims=2, size=(0, 1)):
	m = np.random.uniform(low=size[0], high=size[1], size=(N, dims))
	return m

def simula_gaus(size, sigma, media=None):
	media = 0 if media is None else media
	if len(size) >= 2:
		N = size[0]
		size_sub = size[1:]
		
		out = np.zeros(size, np.float64)
		
		for i in range(N):
			out[i] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=size_sub)
	
	else:
		out = np.random.normal(loc=media, scale=sigma, size=size)
	
	return out

def simula_recta(intervalo=(-1,1), ptos = None):
	if ptos is None:
		m = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
	
	a = (m[0,1]-m[1,1])/(m[0,0]-m[1,0]) # Calculo de la pendiente.
	b = m[0,1] - a*m[0,0]               # Calculo del termino independiente.
	
	return a, b

'''
    Transforma los parámetros de una recta 2d a los coeficientes de w.
    a: Pendiente de la recta.
    b: Término independiente de la recta.
'''
def line2coef(a, b):
	w = np.zeros(3, np.float64)
	w[0] = -a
	w[1] = 1.0
	w[2] = -b
	
	return w


'''
    Pinta los datos con su etiqueta y la recta definida por a y b.
    X: Datos (Intensidad promedio, Simetría).
    y: Etiquetas (-1, 1).
    a: Pendiente de la recta.
    b: Término independiente de la recta.
'''
def plot_datos_recta(X, y, a, b, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
	#Preparar datos
	w = line2coef(a, b)
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01
	
	#Generar grid de predicciones
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],
				   min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = grid.dot(w)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	
	#Plot
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
					   vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$w^tx$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
			cmap="RdYlBu", edgecolor='white', label='Datos')
	ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
	ax.set(
		xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
		ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
		xlabel=xaxis, ylabel=yaxis)
	ax.legend()
	plt.title(title)
	plt.show()


'''
    Pinta los datos con su etiqueta y la recta definida por a y b.
    X: Datos (Intensidad promedio, Simetría).
    y: Etiquetas (-1, 1).
    fz: Devuelve el valor de la altura de la función de etiquetado (z).
    fy: Dada x0, devuelve el valor de x1 para el hiperplano fz=0.
'''
def plot_datos_cuad(X, y, fz, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.001
	#Generar grid de predicciones
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],
				   min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = fz(grid)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	
	#Plot
	f, ax = plt.subplots(figsize=(7,5))
	#ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
	#ax_c = f.colorbar(contour)
	#ax_c.set_label('$f(x, y)$')
	#ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=1,
			cmap="RdYlBu", edgecolor='white', label='Datos')
	#ax.plot(grid[:, 0], fx1(grid[:, 0]), 'black', linewidth=2.0, label='Solucion')
	
	ax.contour(xx,yy,pred_y, [0])
	
	ax.set(
		xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
		ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
		xlabel=xaxis, ylabel=yaxis)
	#ax.legend()
	plt.title(title)
	plt.show()


def ejercicio_11():
	'''
	1. (1 punto) Dibujar una gráfica con la nube de puntos de salida correspondiente.
	'''
	
	'''
		a) Considere N = 50, dim = 2, rango = [−50, +50] con simula_unif (N, dim, rango).
	'''
	X_unif = simula_unif(N=50, dims=2, size=(-50, +50))
	
	'''
		b) Considere N = 50, dim = 2 y sigma = [5, 7] con simula_gaus(N, dim, sigma).
	'''
	X_gauss= simula_gaus((50,2), [5, 7])
	
	
	#Plot data unif
	plt.title('Nube de puntos: distribución uniforme.')
	plt.scatter(X_unif[:, 0], X_unif[:, 1])
	plt.show()
	input("Pulsa enter para continuar.")
	
	#Plot data gauss
	plt.title('Nube de puntos: distribución gaussiana.')
	plt.scatter(X_gauss[:, 0], X_gauss[:, 1])
	plt.show()
	input("Pulsa enter para continuar.")


def label_data(x1,x2,a,b):
	y=np.sign(x2-a*x1-b)
	return y

def modif_aleatorio(y,porcentaje):
	i=0
	y_neg=[]
	y_pos=[]
	for i in range(len(y)):
		if y[i]==-1:
			y_neg.append(i)
		else:
			y_pos.append(i)
	
	idx_neg = np.random.choice(range(len(y_neg)), size=(int(len(y_neg)*(porcentaje/100))), replace=True)
	y[idx_neg] *= -1
	idx_pos = np.random.choice(range(len(y_pos)), size=(int(len(y_pos)*(porcentaje/100))), replace=True)
	y[idx_pos] *= -1
	
	return y

def f1(X):
	return (X[:,0]-10)**2+(X[:,1]-20)**2-400

def f2(X):
	return 0.5*(X[:,0]+10)**2+(X[:,1]-20)**2-400

def f3(X):
	return 0.5*(X[:,0]-10)**2-(X[:,1]+20)**2-400

def f4(X):
	return X[:,1]-20*X[:,0]**2-5*X[:,0]+3

X_global=simula_unif(100,2,(-50,+50))
a_global,b_global=simula_recta((-50,50))
y_global=label_data(X_global[:,0],X_global[:,1],a_global,b_global)

def ejercicio_12_y_13():
	'''
	2. (2 puntos) Con ayuda de la función simula_unif() generar una muestra de puntos 2D a los
	que vamos añadir una etiqueta usando el signo de la función f (x, y) = y − ax − b, es decir
	el signo de la distancia de cada punto a la recta simulada con simula_recta().
	'''
	X=np.copy(X_global)
	a,b=a_global,b_global
	y=np.copy(y_global)
	
	'''
		a) Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto
		con la recta usada para ello. (Observe que todos los puntos están bien clasificados
		respecto de la recta)
	'''
	
	plt.title('Distribución uniforme junto a la función de signo generada por simula_recta')
	plt.axis([-50, 50, -50, 50])
	plt.scatter(X[:, 0], X[:, 1],c=y)
	t = np.arange(-50, 50, 0.1)
	plt.plot(t, a*t+b, 'r-')
	plt.show()
	input("Pulsa enter para continuar.")
	
	'''
		b) Modifique de forma aleatoria un 10 % etiquetas positivas y otro 10 % de negativas
		y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo la gráfica anterior. 
		(Ahora hay puntos mal clasificados respecto de la recta)
	'''
	
	y=modif_aleatorio(y,10)	
	
	plt.title('Distribución uniforme con error en la función de signo.')
	plt.axis([-50, 50, -50, 50])
	plt.scatter(X[:, 0], X[:, 1],c=y)
	t = np.arange(-50, 50, 0.1)
	plt.plot(t, a*t+b, 'r-')
	plt.show()
	input("Pulsa enter para continuar.")
	
	'''
	3. (3 puntos) EJERCICIO 3:
	'''
	
	plot_datos_cuad(X,y,f1,title='Distribución uniforme con error y f(x,y)=(x − 10)² + (y − 20)² − 400')
	input("Pulsa enter para continuar.")
	
	plot_datos_cuad(X,y,f2,title='Distribución uniforme con error y f(x,y)=0,5(x + 10)² + (y − 20)² − 400')
	input("Pulsa enter para continuar.")
	
	plot_datos_cuad(X,y,f3,title='Distribución uniforme con error y f(x,y)=0,5(x − 10)² − (y + 20)² − 400')
	input("Pulsa enter para continuar.")
	
	plot_datos_cuad(X,y,f4,title='Distribución uniforme con error y f(x,y)=y − 20x² − 5x + 3')
	input("Pulsa enter para continuar.")
		


def ajusta_PLA(datos, label, max_iter, wini):
	iteraciones=0
	wold=np.copy(wini)
		
	while iteraciones < max_iter:
		i=0
		for i in range(len(datos)):
			if np.sign(np.dot(np.transpose(wini),datos[i])) != label[i]:
				wini=wini+np.dot(label[i],datos[i])
				
		
		if np.array_equal(wini,wold):
			break
		
		wold=np.copy(wini)
		
		iteraciones+=1 
	
	return wini,iteraciones

def aniade_coordenada(X):
	Xtemp=[]
	
	i=0
	for i in range(len(X)):
		Xtemp.insert(len(Xtemp),[1,X[i][0],X[i][1]])
	
	solX=np.array(Xtemp, np.float64)
	
	return solX

def ejercicio_21():	
	X=np.copy(X_global)
	X=aniade_coordenada(X)
	a,b=a_global,b_global
	y=np.copy(y_global)
	
	'''
	a) Ejecutar el algoritmo PLA con los datos simulados en los apartados 2a de la sección.1.
	Inicializar el algoritmo con: a) el vector cero y, b) con vectores de números aleatorios
	en [0, 1] (10 veces). Anotar el número medio de iteraciones necesarias en ambos para
	converger. Valorar el resultado relacionando el punto de inicio con el número de
	iteraciones.
	'''
	
	#OMEGA CERO
	wini1=[0,0,0]
	w1,it1=ajusta_PLA(X,y,10000,wini1)
	plt.axis([-50, 50, -50, 50])
	t = np.arange(-50, 50, 0.1)
	plt.plot(t,-w1[0]/w1[2] - w1[1]/w1[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
	plt.plot(t, a*t+b, 'b-')
	plt.scatter(X[:, 1],X[:, 2], c=y)
	plt.title('Ajuste de los datos mediante PLA (w=0).')
	plt.show()
	input("Pulsa enter para continuar.")
	
	#OMEGA ALEATORIO
	wini2=np.random.rand(3)
	w2,it2=ajusta_PLA(X,y,10000,wini2)
	plt.axis([-50, 50, -50, 50])
	t = np.arange(-50, 50, 0.1)
	plt.plot(t,-w2[0]/w2[2] - w2[1]/w2[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
	plt.plot(t, a*t+b, 'b-')
	plt.scatter(X[:, 1],X[:, 2], c=y)
	plt.title('Ajuste de los datos mediante PLA (w aleatoria).')
	plt.show()
	input("Pulsa enter para continuar.")
	print(' ')
	
	print('DATOS SIN MODIFICAR')
	print('Valor e iteraciones con ', wini1)
	print(w1)
	print(it1)
	print(' ')
	print('Valor e iteraciones con ', wini2)
	print(w2)
	print(it2)
	print(' ')
	
	
	media_it2=it2
	for i in range(1,10):
		wini2=np.random.rand(3)
		basura,tmp=ajusta_PLA(X,y,10000,wini2)
		#print(tmp)
		media_it2+=tmp
		
	print('Media de iteraciones para w aleatorio: ', media_it2/10)
	
	input("Pulsa enter para continuar.")
	
	
	'''
	b) Hacer lo mismo que antes usando ahora los datos del apartado 2b de la sección.1.
	¿Observa algún comportamiento diferente? En caso afirmativo diga cual y las razones
	para que ello ocurra.
	'''
	y=modif_aleatorio(y,10)
	
	#OMEGA CERO
	wini1=[0,0,0]
	w1,it1=ajusta_PLA(X,y,10000,wini1)
	plt.axis([-50, 50, -50, 50])
	t = np.arange(-50, 50, 0.1)
	plt.plot(t,-w1[0]/w1[2] - w1[1]/w1[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
	plt.plot(t, a*t+b, 'b-')
	plt.scatter(X[:, 1],X[:, 2], c=y)
	plt.title('Ajuste de los datos modificados mediante PLA (w=0).')
	plt.show()
	input("Pulsa enter para continuar.")
	
	#OMEGA ALEATORIO
	wini2=np.random.rand(3)
	w2,it2=ajusta_PLA(X,y,10000,wini2)
	plt.axis([-50, 50, -50, 50])
	t = np.arange(-50, 50, 0.1)
	plt.plot(t,-w2[0]/w2[2] - w2[1]/w2[2]*t, 'r-') # La fórmula de la recta la obtenemos por h(x)=0
	plt.plot(t, a*t+b, 'b-')
	plt.scatter(X[:, 1],X[:, 2], c=y)
	plt.title('Ajuste de los datos modificados mediante PLA (w aleatoria).')
	plt.show()
	input("Pulsa enter para continuar.")
	print(' ')
	
	print('DATOS MODIFICADOS')
	print('Valor e iteraciones con ', wini1)
	print(w1)
	print(it1)
	print(' ')
	print('Valor e iteraciones con ', wini2)
	print(w2)
	print(it2)
	print(' ')
	input("Pulsa enter para continuar.")



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

def sigma(y,w,X):	
	omega_x=np.dot(np.transpose(w),X)
	return 1/(1+np.exp(-np.dot(-y,omega_x)))

def reetiqueta(y): #Cambia los -1 del vecto y por 0
	i=0
	for i in range(len(y)):
		if y[i]==-1:
			y[i]=0
	return y

def regresion_logistica_SGD(X_,y_,wini,lr,epsilon,longitud,max_iter):	
	#w = np.zeros(len(X_[1]),np.float64)
	w=np.copy(wini)
	
	X, y=set_minibatches(X_,y_,longitud)
	
	wold=np.copy(w)
	
	i=0
	while i < max_iter:
		suma=np.zeros(len(X[1]), np.float64)
		
		j=0
		for j in range(len(X)):
			suma += np.dot(np.dot(-y[j],X[j]), sigma(y[j],w,X[j]))
		
		w=wold-lr*suma
		
		if np.linalg.norm(wold-w) < epsilon:
			break
	
		
		wold=np.copy(w)
		
		X, y=set_minibatches(X_,y_,longitud)
		i+=1
	
	print('Iteraciones necesarias:',i)
	
	return w

def tasa_error(X,y,w):
	total=len(X)
	fallos=0
	i=0
	
	for i in range(total):
		omega_t=np.dot(np.transpose(w),X[i])
		valor=1/(1+np.exp(-omega_t))
		
		if valor>0.5 and y[i]==0:
			fallos+=1
		if valor<0.5 and y[i]==1:
			fallos+=1
		
	porcentaje_fallo=1.0*fallos/total
	
	return porcentaje_fallo


def error_out(X,y,w):
	error=0
	suma=0
	#y=reetiqueta(y)
	
	i=0
	for i in range(len(X)):
		exponente=np.dot(-y[i], np.dot(np.transpose(w),X[i]))
		suma+=np.log(1+np.exp(exponente))
	
	error=suma*1.0/len(X)
	return error
	
def ejercicio_22():
	'''
	2. (4 puntos) Regresión Logística: [...] 	Consideremos d = 2 para que los datos sean 
	visualizables, y sea X = [0, 2] × [0, 2] con probabilidad uniforme de elegir cada x ∈ X .
	Elegir una línea en el plano que pase por X como la frontera entre f(x)=1 (donde y
	toma valores +1) y f(x)=0 (donde y toma valores −1), para ello seleccionar dos puntos
	aleatorios del plano y calcular la línea que pasa por ambos. Seleccionar N = 100 puntos
	aleatorios {x_n} de X y evaluar las respuestas {y_n} de todos ellos respecto de la
	frontera elegida.
	'''
	
	X=simula_unif(100,2,(0,2))
	a,b=simula_recta((0,2))
	y=label_data(X[:,0],X[:,1],a,b)
	wini=[0,0,0]
	
	'''
		b) Usar la muestra de datos etiquetada para encontrar nuestra solución g ... 
	'''
	X=aniade_coordenada(X)
	#y=reetiqueta(y)
	
	w=regresion_logistica_SGD(X,y,wini,0.01,0.01,64,10000)
	
	plt.axis([0, 2, 0, 2])
	t = np.arange(-5, 5, 0.1)
	plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t, 'r-')
	plt.plot(t, a*t+b, 'g-')
	plt.scatter(X[:, 1],X[:, 2], c=y)
	plt.title('Ajuste mediante Regresión Logística con SGD.')
	plt.show()
	input("Pulsa enter para continuar.")
	
	'''
		... y estimar E out usando para ello un número suficientemente grande de nuevas
		muestras (>999).
	'''
	
	X=simula_unif(5000,2,(0,2))
	y=label_data(X[:,0],X[:,1],a,b)
	X=aniade_coordenada(X)
	
	plt.axis([0, 2, 0, 2])
	t = np.arange(-5, 5, 0.1)
	plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t, 'r-')
	#plt.plot(t, a*t+b, 'g-')
	plt.scatter(X[:, 1],X[:, 2], c=y)
	plt.title('Ajuste mediante Regresión Logística con SGD.')
	plt.show()
	input("Pulsa enter para continuar.")
	print('')
	
	e=error_out(X,y,w)
	print('El error de salida con 5000 nuevas muestras generadas será de: ',e)
	
	y=reetiqueta(y)
	
	tasa=tasa_error(X,y,w)	
	
	print('La tasa de error respecto de la sigmoide será de: ',tasa)

def main():
	ejercicio_11()
	
	ejercicio_12_y_13()
	
	ejercicio_21()
	
	ejercicio_22()
	
	print(' ')
	print('FIN')
	
if __name__ == '__main__':
	main()


