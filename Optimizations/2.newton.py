'''
This code is an example for one dimentional newton method

Author: Dietrich Klakow based
'''
#Project: https://repos.lsv.uni-saarland.de/dietrich/Neural_Networks_Implementation_and_Application/tree/master

import math
from math import pow

import numpy as np
import numpy.linalg as lin
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f( x, y ):
	return 3 * pow(x, 2) - pow(y, 2)

def F( x, y ):
	return 3 * x.dot(x) - y.dot(y)

def dfdx( x, y ):
	return 6 * x

def dfdy( x, y ):
	return -2 * y

def dfdxdx( x, y ):
	return 6

def dfdxdy( x, y ):
	# using small value instead of 0
	return 1e-10

def dfdydx( x, y ):
	# using small value instead of 0
	return 1e-10

def dfdydy( x, y ):
	return -2

# multiplier for 1st derivative
fpm = np.array([6, -2], dtype='float')

# using small value instead of 0
def Hessian(x, y):
	return np.array([
	[dfdxdx(x, y), dfdxdy(x, y)], 
	[dfdydx(x, y), dfdydy(x, y)]])


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
v=np.array([5, 1], dtype='float')

print v
print fpm
print Hessian(v[0], v[1])

#Learning rate
epsilon=0.1
converge=False

x_list = list()
y_list = list()
i = 1
max_iter = 50
sc = 0

while not converge and i < max_iter:
	fo=f(v[0], v[1])
	i += 1
	x_list.append(v[0])
	y_list.append(v[1])
	fp = np.multiply(v, fpm)

	v = v - epsilon * np.dot(fp, lin.inv(Hessian(v[0], v[1])))
	sc = math.fabs(fo - f(v[0], v[1]))
	if ( sc < 1e-3):
		converge=True
	print i, v, fo, fp, sc


plotContour(x_list, y_list, title="7.2.e Newton Method", file_name="7.2.e.png")

plot3D(x_list, y_list, title="7.2.e 3D of Newton Method", file_name="7.2.Newton.3D.png", toShow=True)
