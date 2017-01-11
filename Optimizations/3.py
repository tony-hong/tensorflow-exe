'''
This code is an example for one dimentional gradient descent

Author: Dietrich Klakow based
		Tony Hong (modified)
'''
#Project: https://repos.lsv.uni-saarland.de/dietrich/Neural_Networks_Implementation_and_Application/tree/master

import math
from math import pow

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f( x, y ):
	return 1e-3 * pow(x, 2) - 1e-3 * pow(y, 2)

def dfdx( x, y ):
	return 2e-3 * x

def dfdy( x, y ):
	return -2e-3 * y

def df(x):
	return np.array([2e-3, -2e-3]) * x

def gd_update(x, y, epsilon):
	x = x - epsilon * dfdx(x, y)
	y = y - epsilon * dfdy(x, y)
	return x, y

def adaGrad_update(x, y, g):
	x = x + g[0]
	y = y + g[1]
	return x, y

def converged(x, x_old, standard=1e-8):
	return math.fabs(x - x_old) < standard

def gd(x_init, y_init, max_iter, epsilon=0.01, alpha=0.7, wMomentum=False):
	converge=False

	x = x_init
	y = y_init
	x_list = list()
	y_list = list()
	i = 1
	v_x = epsilon
	v_y = epsilon
	print('epsilon', epsilon)

	while not converge and i < max_iter:
		fo = f(x, y)
		i += 1
		print i, x, y, fo
		x_list.append(x)
		y_list.append(y)
		
		if wMomentum:
			v_x = alpha * v_x - epsilon * dfdx(x, y)
			v_y = alpha * v_y - epsilon * dfdy(x, y)
			x = x + v_x
			y = y + v_y
		else:
			x, y = gd_update(x, y, epsilon)
		sc = math.fabs(fo-f(x, y))
		if ( sc < 1e-10):
			converge=True
	return x_list, y_list

def adaGrad(x_init, y_init, max_iter, epsilon=0.01, delta=1e-7):
	converge=False

	x = x_init
	y = y_init
	x_list = list()
	y_list = list()
	i = 1
	r = 0
	print('epsilon', epsilon)

	while not converge and i < max_iter:
		fo = f(x, y)
		i += 1
		print i, x, y, fo
		x_list.append(x)
		y_list.append(y)
		
		g = df(np.array([x, y]))
		r = r + g.dot(g)
		theta = - epsilon / (delta + math.sqrt(r)) * g
		x, y = adaGrad_update(x, y, g)
		sc = math.fabs(fo-f(x, y))
		if ( sc < 1e-10):
			converge=True
	return x_list, y_list



def plotContour(x_list, y_list, title="Data Graph", file_name="data_graph", delta=0.01, bound=5, toShow=False):
	X = np.arange(-bound, bound, delta)
	Y = np.arange(-bound, bound, delta)
	Z = [[f(x, y) for x in X] for y in Y]
	# print F

	plt.figure()
	ct = plt.contour(X, Y, Z)
	plt.plot(x_list, y_list)
	plt.clabel(ct, inline=1, fontsize=10)

	plt.xlabel('x')
	plt.ylabel('y')

	plt.title(title)
	plt.savefig(file_name)
	if toShow:
		plt.show()

def plot3D(x_list, y_list, title="Data Graph", file_name="data_graph", delta=0.1, bound=5, toShow=False):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	X = np.arange(-bound, bound, delta)
	Y = np.arange(-bound, bound, delta)

	Z = [[f(x, y) for x in X] for y in Y]

	X = np.tile(X, [len(X), 1])
	Y = X.T
	# Y = np.tile(Y.reshape(l_Y, 1), [1, l_Y])

	R = list(x_list)
	for i in range(len(x_list)):
		R[i] = f(x_list[i], y_list[i])

	ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

	ax.scatter(x_list, y_list, R, c='r')

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('f=0.001x^2 - 0.001y^2')

	plt.title(title)
	plt.savefig(file_name)
	if toShow:
		plt.show()



# Initial point
x=3
y=-1

#Learning rate
epsilon=0.1

max_iter = 300

alpha = 0.7

Delta = 0.01
bound = 5

x_list, y_list = gd(x, y, max_iter, epsilon, alpha, wMomentum=False)

plotContour(x_list, y_list, title="7.3.a. GD", file_name="7.3.a.png")

plot3D(x_list, y_list, title="7.3.c. 3D of GD", file_name="7.3.GD.3D.png", toShow=True)

x=3
y=-1

x_list, y_list = adaGrad(x, y, max_iter, epsilon, delta=1e-9)

plotContour(x_list, y_list, title="7.3.b. AdaGrad", file_name="7.3.b.png")

plot3D(x_list, y_list, title="7.3.c. 3D of AdaGrad", file_name="7.3.AdaGrad.3D.png", toShow=True)
