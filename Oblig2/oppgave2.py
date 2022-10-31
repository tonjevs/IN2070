from cmath import sqrt
from imageio import imread,imsave
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import jit
from scipy.fftpack import dct, idct
import math

def main():
    filename = 'uio.png'

    f = imread(filename,as_gray = True)
    f3,en1 = kompresjon(f,0.1)
    f4,en2 = kompresjon(f,0.5)
    f5,en3 = kompresjon(f,2)
    f6,en4 = kompresjon(f,8)
    f7,en5 = kompresjon(f,32)

    print("q=0.1",entropi(en1),'\n',"q=0.5",entropi(en2),'\n',"q=2",entropi(en3),'\n',"q=8",entropi(en4),'\n',"q=32",entropi(en5))

    verdi = f - f3
    print(verdi)#Om denne arrayen var 0, hadde det vert samme bilde

    plt.subplot(2,3,1)
    plt.imshow(f,cmap='gray')
    plt.title("Original")
    plt.subplot(2,3,2)
    plt.imshow(f3,cmap='gray')
    plt.title("q=0.1")
    plt.subplot(2,3,3)
    plt.imshow(f4,cmap='gray')
    plt.title("q=0.5")
    plt.subplot(2,3,4)
    plt.imshow(f5,cmap='gray')
    plt.title("q=2")
    plt.subplot(2,3,5)
    plt.imshow(f6,cmap='gray')
    plt.title("q=8")
    plt.subplot(2,3,6)
    plt.imshow(f7,cmap='gray')
    plt.title("q=32")
    plt.show()

    imsave("0.1.uio.jpeg",f3)
    imsave("0.5.uio.jpeg",f4)
    imsave("2.uio.jpeg",f5)
    imsave("8.uio.jpeg",f6)
    imsave("32.uio.jpeg",f7)

@jit
def kompresjon(f,q):

    Q = np.array([[16,11,10,16,24,40,51,61],
                  [12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56],
                  [14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],
                  [24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]])

    N,M = f.shape

    f = f - 128

    f2 = np.zeros((N,M))

    sum = 0
    for a in range(0,N,8):
        for b in range(0,M,8):
            for u in range(8):
                for v in range(8):
                    for x in range(8):
                        for y in range(8):
                            sum += (f[x+a,y+b]*(np.cos(((2*x + 1)*u*np.pi)/16))*(np.cos(((2*y + 1)*v*np.pi)/16)))
                    f2[a+u,b+v] = np.round((1/4 * c(u) * c(v) * sum / (q*Q[u,v])))
                    sum = 0

    f3 = np.zeros((N,M))

    for a in range(0,N,8):
        for b in range(0,M,8):
            for u in range(8):
                for v in range(8):
                    for x in range(8):
                        for y in range(8):
                            sum += (c(x) * c(y) * f2[x+a,y+b]*(np.cos(((2*u + 1)*x*np.pi)/16))*(np.cos(((2*v + 1)*y*np.pi)/16))) * (q*Q[x,y]) 
                    f3[a+u,b+v] = np.round(1/4 * sum)
                    sum = 0
    f3 = f3 + 128

    return f3,f2

@jit
def c(a):
    if a == 0: return 1/np.sqrt(2)
    else: 
        return 1

def entropi(f):
    hist_ord = {}

    A,B = f.shape  
    N = A*B

    for i in range(A):
        for j in range(B):
            if f[i,j] not in hist_ord:
                hist_ord[f[i,j]] = 1
            else:
                verdi = hist_ord.get(f[i,j]) + 1
                hist_ord[f[i,j]] = verdi

    H = 0

    for x in hist_ord:
        verdi = hist_ord.get(x)/N
        H += verdi * np.log2(verdi)

    H = -H
    return H

if __name__ == "__main__":
    main()