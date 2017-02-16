'''
Power Method Test

Author: Xi Chen
'''

import matplotlib.pylab as plt
import numpy as np

def power_method(mat, start):
    result = start
    i = 1
    dist = check(mat,result)
    pl = [10]
    x  = [10]
    x[0] = 1
    pl[0] = dist
    while(dist > 1e-4):
            result = mat*result
            result = result/np.linalg.norm(result)
            dist = check(mat,result)
            pl.append(dist)
            i = i+1
            x.append(i)
    plt.plot(x,pl)
    plt.show()
    return result

def check(mat, otp):

    prd = mat*otp
    eigval = prd[0]/otp[0]
    print 'computed eigenvalue :' , eigval
    [eigs, vecs] = np.linalg.eig(mat)
    abseigs = list(abs(eigs))
    ind = abseigs.index(max(abseigs))
    print ' largest eigenvalue :', eigs[ind]
    dist = np.sqrt(np.sum(np.square(eigval - eigs[ind])))
    print 'distance    ', dist
    return dist

def main():

    print 'Running the power method...'
    dim = 3
    nbs = np.random.normal(0, 1, (dim, 1)) 
    mat = np.matrix('-2 -2 3;-10 -1 6; 10 -2 -9')
    rndvec = np.matrix(nbs)
    eigmax = power_method(mat, rndvec)
    
main()