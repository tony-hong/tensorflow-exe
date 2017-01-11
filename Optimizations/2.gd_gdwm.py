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
	return 3 * pow(x, 2) - pow(y, 2)

def F( X, Y ):
	return 3 * X.dot(X) - Y.dot(Y)

def dfdx( x, y ):
	return 6 * x

def dfdy( x, y ):
	return -2 * y

def df(x):
	return np.array([6, -2]) * x

def gd_update(x, y, epsilon):
	x = x - epsilon * dfdx(x, y)
	y = y - epsilon * dfdy(x, y)
	return x, y

def gd(x_init, y_init, epsilon=0.01, alpha=0.7, wMomentum=False):
	converge=False

	x = x_init
	y = y_init
	x_list = list()
	y_list = list()
	i = 1
	v_x = epsilon
	v_y = epsilon

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
		if ( sc < 1e-3):
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
	ax.set_zlabel('f=3x^2-y^2')

	plt.title(title)
	plt.savefig(file_name)
	if toShow:
		plt.show()


# Initial point
x=5
y=-1

#Learning rate
epsilon=0.01

max_iter = 30

alpha = 0.7

delta = 0.01
bound = 5


# 7.2.a. GD
x_list, y_list = gd(x, y, epsilon=0.01, alpha=0.7, wMomentum=False)

plotContour(x_list, y_list, title="7.2.a. GD", file_name="7.2.a.png", toShow=False)

plot3D(x_list, y_list, title="7.2.c. 3D of GD", file_name="7.2.GD.3D.png", toShow=True)

# 7.2.b. GD with momentum
xlist, y_list = gd(x, y, epsilon=0.01, alpha=0.7, wMomentum=True)

plotContour(x_list, y_list, title="7.2.b. GD with momentum", file_name="7.2.b.png", toShow=False)

plot3D(x_list, y_list, title="7.2.c. 3D of GD with momentum", file_name="7.2.GDwM.3D.png", toShow=True)
