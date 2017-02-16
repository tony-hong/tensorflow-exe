'''
Garbor Filter Test

A test of Garbor Filter which is similar to 1st layer of CNN.

Author: Xi Chen
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
def build_filters():
 filters = []
 ksize = 10
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 3.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
 accum = np.zeros_like(img)
 for kern in filters:
     fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
     #np.maximum(accum, fimg, accum)
 return fimg
 
if __name__ == '__main__':
 import sys

 print __doc__
 try:
     img_fn = sys.argv[1]
 except:
     img_fn = 'hide.png'
 
 img = cv2.imread(img_fn)
 if img is None:
     print 'Failed to load image file:', img_fn
     sys.exit(1)
 
 filters = build_filters()
 res1 = process(img, filters)
 plt.figure(1)
 plt.subplot(121)
 plt.imshow(img)
 plt.subplot(122)
 plt.imshow(res1)
 plt.show()
