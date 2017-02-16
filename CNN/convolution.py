'''
Convolution test.

A test of convolution layer.

Author: Xi Chen
'''
from __future__ import division
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

def readImage(path):
	image = misc.imread(path)
	return image
	
def conv(image, kernel):
	shape = image.shape
	ix = shape[0]
	iy = shape[1]
	res = np.empty(shape=[ix,iy])
	for i in range(ix):
		for j in range(iy):
			for m in range(0,2):
				for n in range(0,2):
					res[i][j] += (kernel[m][n] * image[i-m][j-n])
	shape2 = res.shape
	sx = shape2[0]
	sy = shape2[1]
	res = np.pad(res,pad_width = (ix-sx,iy-sy), mode = 'constant', constant_values = 0)
	return res
	
def createKern(num):
	kernel = np.empty(shape = [3,3])
	if num == 1:
		kernel = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
	else :
		kernel = np.array([[0,0,0],[1,-2,1],[0,0,0]])
	return kernel
	
def min_max_rescale(image):
    s = image.shape
    ix = s[0]
    iy = s[1]

    maxV = image.max()
    minV = image.min()

    print minV, maxV

    res = np.zeros((ix,iy))

    rangeV = maxV - minV

    for x in range(ix):
        for y in range(iy):
            res[x][y] = (image[x][y] - minV) / float(rangeV) * 255

    return res

def main():
# 8.2.b
   path = 'clock_noise.png'
   im = readImage(path)
   plt.figure(1)
   plt.subplot(121)
   plt.imshow(im, cmap = plt.cm.gray)
   kernel_1 = createKern(1)
   res_first = conv(im,kernel_1)
   res = min_max_rescale(res_first)
   plt.subplot(122)
   plt.imshow(res, cmap = plt.cm.gray)
   plt.show()
# 8.2.c
   path = 'clock.png'
   im2 = readImage(path)
   kernel_2 = createKern(2)
   res_first = conv(im2,kernel_2)
   res = min_max_rescale(res_first)
   plt.figure(2)
   plt.subplot(121)
   plt.imshow(im2, cmap = plt.cm.gray)
   plt.subplot(122)
   plt.imshow(res, cmap = plt.cm.gray)
   plt.show()
   
   
if __name__=="__main__":
    main()
