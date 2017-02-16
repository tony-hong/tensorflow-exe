'''
Pooling Test

A test of max pooling and mean pooling.

Author: 
    Xi Chen
    Tony Hong
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
	res = np.empty(shape=[ix,iy], dtype='float')
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
	kernel = np.empty(shape = [3,3], dtype='float')
	if num == 1:
		kernel = np.array([
      [1/9,1/9,1/9],
      [1/9,1/9,1/9],
      [1/9,1/9,1/9]], dtype='float')
	else :
		kernel = np.array([
      [0,0,0],
      [1,-2,1],
      [0,0,0]], dtype='float')
	return kernel
	
def min_max_rescale(image):
    s = image.shape
    ix = s[0]
    iy = s[1]

    maxV = image.max()
    minV = image.min()

    print minV, maxV

    res = np.zeros((ix,iy), dtype='float')

    rangeV = maxV - minV

    for x in range(ix):
        for y in range(iy):
            res[x][y] = (image[x][y] - minV) / float(rangeV) * 255

    return res


def maxPool(data, split):
  l = data.shape[0]
  step = int(l / split)
  res = np.zeros([split, split], dtype='float')
  for i in range(split):
    for j in range(split):
      lower_i = step * i
      higher_i = (step * (i + 1) - 1)
      lower_j = step * j
      higher_j = (step * (j + 1) - 1)
      res[i][j] = np.max(data[lower_i : higher_i, lower_j : higher_j])
  return res


def meanPool(data, split):
  l = data.shape[0]
  step = int(l / split)
  res = np.zeros([split, split], dtype='float')
  for i in range(split):
    for j in range(split):
      lower_i = step * i
      higher_i = (step * (i + 1) - 1)
      lower_j = step * j
      higher_j = (step * (j + 1) - 1)
      res[i][j] = np.mean(data[lower_i : higher_i, lower_j : higher_j])
  return res


def exe8():
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


def exe9_4_b(im_org):
  result_dict = dict()
  max_dict = dict()
  max_pos_dict = dict()
  for i in [8, 4, 2, 1]:
    result_dict[i] = maxPool(im_org, i)
    max_dict[i] = np.max(result_dict[i])
    pos = np.argmax(result_dict[i])
    px = int(pos / i)
    py = int(pos % i)
    max_pos_dict[i] = (pos, px, py)

  print max_pos_dict

  plt.figure(1)
  plt.subplot(231)
  plt.imshow(im_org, cmap = plt.cm.gray)
  plt.subplot(233)
  plt.imshow(result_dict[8], cmap = plt.cm.gray)
  plt.subplot(234)
  plt.imshow(result_dict[4], cmap = plt.cm.gray)
  plt.subplot(235)
  plt.imshow(result_dict[2], cmap = plt.cm.gray)
  plt.subplot(236)
  plt.imshow(result_dict[1], cmap = plt.cm.gray)
  plt.savefig('9.4.b.maxPool.png')
  plt.show()


def exe9_4_c(im_org):
  split = 128
  meanPool_out = meanPool(im_org, split)
  maxPool_out = maxPool(im_org, split)

  plt.figure(1)
  plt.imshow(meanPool_out, cmap = plt.cm.gray)
  plt.savefig('clockMean.png')
  plt.show()

  plt.figure(2)
  plt.imshow(maxPool_out, cmap = plt.cm.gray)
  plt.savefig('clockMax.png')
  plt.show()

  plt.figure(3)
  plt.subplot(131)
  plt.imshow(im_org, cmap = plt.cm.gray)
  plt.subplot(132)
  plt.imshow(meanPool_out, cmap = plt.cm.gray)
  plt.subplot(133)
  plt.imshow(maxPool_out, cmap = plt.cm.gray)
  plt.savefig('9.4.c.maxPool_vs_meanPool_split_' + str(split) + '.png')
  plt.show()

  split = 32
  meanPool_out = meanPool(im_org, split)
  maxPool_out = maxPool(im_org, split)

  plt.figure(4)
  plt.subplot(131)
  plt.imshow(im_org, cmap = plt.cm.gray)
  plt.subplot(132)
  plt.imshow(meanPool_out, cmap = plt.cm.gray)
  plt.subplot(133)
  plt.imshow(maxPool_out, cmap = plt.cm.gray)
  plt.savefig('9.4.c.maxPool_vs_meanPool_split_' + str(split) + '.png')
  plt.show()


def main():
  path = 'clock.png'
  im_org = readImage(path)

# 9.4.b
  exe9_4_b(im_org)

# 9.4.b
  exe9_4_c(im_org)
  


if __name__=="__main__":
    main()
