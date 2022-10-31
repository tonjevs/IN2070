from os import lseek
from imageio import imread
import matplotlib.pyplot as plt
from scipy import signal
import time
import numpy as np
from numba import jit

def main():
    filename = 'cow.png'

    f = imread(filename,as_gray = True)
    N,M = f.shape

    filter = np.ones((15,15)) * 1/(15**2)
    c = signal.convolve2d(f,filter)

    y = np.fft.fft2(f)
    a = padding(filter,f)
    b = np.fft.fft2(a)

    multi = y * b
    for_shift = np.fft.fft2(multi)
    etter_shift = np.fft.fftshift(for_shift)
    etter_shift = np.real(etter_shift)
    etter_shift = np.rot90(etter_shift)
    etter_shift = np.rot90(etter_shift)

    y3 = signal.convolve2d(f,filter,mode = 'same')
    op2 = np.real(y3)

    y = np.fft.fft2(f)
    b = np.fft.fft2(filter,((N,M)))

    etter_shift1 = np.fft.fftshift(y)
    etter_shift2 = np.fft.fftshift(b)

    ganget = etter_shift1 * etter_shift2
    hei = np.fft.ifft2(ganget)

    etter_shift3 = np.abs(hei)

    plt.subplot(2,3,1)
    plt.imshow(f,cmap='gray')
    plt.title("Startbilde")
    plt.subplot(2,3,2)
    plt.imshow(c,cmap='gray')
    plt.title("Konvolvert")   
    plt.subplot(2,3,3)
    plt.imshow(etter_shift,cmap='gray')
    plt.title("Siste metode")
    plt.subplot(2,3,4)
    plt.imshow(op2,cmap='gray')
    plt.title("Oppg 1.2")
    plt.subplot(2,3,5)
    plt.imshow(etter_shift3,cmap='gray')
    plt.title("Oppg 1.2 andre")  
    plt.show()

    tidsutregning(f)

def padding(filter,f):
    width,length = f.shape
    a,b = filter.shape

    if(a % 2 != 0):
        add = 0
    elif(a % 2 == 0):
        add = 1

    amount = int((width - a)/2)
    amount2 = int((length - b)/2)

    nmatrix = np.zeros((a,b))
    c = [nmatrix[0]]
    
    for i in range(amount-add):
        c = np.append(c,[nmatrix[0]],axis=0)

    filter = np.append(c,filter,axis=0)
    
    for i in range(amount):
        filter = np.append(filter,[nmatrix[-1]],axis=0)
  
    filter = np.transpose(filter)
    a,b = filter.shape
    nmatrix = np.zeros((a,b))
    c = [nmatrix[0]]
    
    for i in range(amount2-add):
        c = np.append(c,[nmatrix[0]],axis=0)

    filter = np.append(c,filter,axis=0)
    
    for i in range(amount2):
        filter = np.append(filter,[nmatrix[-1]],axis=0)
    
    filter = np.transpose(filter)
    
    return filter

def tidsutregning(f):
    dicts = []
    dicts.append(0)
    dict2 = []
    dict2.append(0)
    N,M = f.shape

    plt.figure()
    s = [i for i in range(1,30+1)]
    for i in range(1,30):
        start_tid = time.time()
        filter = np.ones((i,i)) * 1/(i**2)
        c = signal.convolve2d(f,filter)
        start_tid = time.time() - start_tid
        dicts.append(start_tid)

        start_tid = time.time()

        y = np.fft.fft2(f)
        b = np.fft.fft2(filter,((N,M)))

        etter_shift1 = np.fft.fftshift(y)
        etter_shift2 = np.fft.fftshift(b)

        ganget = etter_shift1 * etter_shift2
        hei = np.fft.ifft2(ganget)

        etter_shift3 = np.abs(hei)

        start_tid = time.time() - start_tid
        dict2.append(start_tid)
        
    plt.plot(s,dicts,color='b')
    plt.plot(s,dict2,color='g')
    plt.title("Graf over kjøretid") 
    plt.show()

if __name__ == "__main__":
    main()

