from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

def main():
    
    filename = 'cellekjerner.png'
    filename_2 = 'detekterte_kanter.png'
    f = imread(filename,as_gray = True)
    los = imread(filename_2,as_gray = True)
    plt.imshow(f,cmap='gray')
    plt.show()
    N,M = f.shape

    filter = gaussian(3.6)
    a = padding(f,filter)
    y = adder_padding(f,a,filter)
    x = find_gradient(y)
    z = thinning(x[0],x[1])
    hyst = hysteresis(z,60,30)
    print(hyst)

    plt.subplot(2,3,1)
    plt.imshow(f,cmap='gray')
    plt.title("Startbilde")
    plt.subplot(2,3,2)
    plt.imshow(y,cmap='gray')
    plt.title("Gauss Filter")
    plt.subplot(2,3,3)
    plt.imshow(x[0],cmap='gray')
    plt.title("Gradient Magnitude")
    plt.subplot(2,3,4)
    plt.imshow(z,cmap='gray')
    plt.title("Tynnet bilde")
    plt.subplot(2,3,5)
    plt.imshow(hyst,cmap='gray')
    plt.title("Hysterese")
    plt.subplot(2,3,6)
    plt.imshow(los,cmap='gray')
    plt.title("Fasit")
    plt.show()

@jit
def padding(f,filter):
    N,M = f.shape
    a,b = filter.shape
    width = int(np.floor(a/2))
    length = int(np.floor(b/2))
    c = [f[0]]
    
    for i in range(width-1):
        c = np.append(c,[f[0]],axis=0)

    f = np.append(c,f,axis=0)
    
    for i in range(width):
        f = np.append(f,[f[-1]],axis=0)
  
    f = np.transpose(f)
    c = [f[0]]
    
    for i in range(length-1):
        c = np.append(c,[f[0]],axis=0)

    f = np.append(c,f,axis=0)
    
    for i in range(length):
        c = np.append(f,[f[-1]],axis=0)
    
    f = np.transpose(c)
    
    return f
@jit
def adder_padding(f_org,f,filter):

    A,B = f_org.shape
    N,M = f.shape
    a,b = filter.shape
    filter = np.flipud(np.fliplr(filter))
    g = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            sum = 0

            for k in range(a):
                for l in range(b):
                    if i+k in range(N) and j+l in range(M):
                        sum += f[i+k,j+l]*filter[k,l]
                    else:
                        sum += 0
            g[i,j]= sum

    g = g[0:A,0:B]

    return g
@jit
def gaussian(sigma):
    length = np.ceil(1 + 8*sigma) / 2
    x, y = np.mgrid[-length:length+1, -length:length+1]
    A = 1 / (2 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2*sigma**2))) * A
    return g

@jit
def find_gradient(f):

    sobel_1 = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])

    sobel_2 = np.array([[1,2,1],
                       [0,0,0],
                       [-1,-2,-1]])

    I_x = adder_padding(f,padding(f,sobel_1),sobel_1)
    I_y = adder_padding(f,padding(f,sobel_2),sobel_2)
    G = np.sqrt(I_x**2 + I_y**2)
    theta = np.arctan2(I_y,I_x)
    return (G, theta)

def thinning(f,theta):
    N,M = f.shape
    angle = np.zeros((N,M))
    Z = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            angle[i,j] = round(theta[i,j] * (180 / np.pi) * (1 / 45) * 45) % 180

    for i in range(1,N-1):
        for j in range(1,M-1):
            q = 255
            r = 255
                
           #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = f[i, j+1]
                r = f[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = f[i+1, j-1]
                r = f[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = f[i+1, j]
                r = f[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = f[i-1, j-1]
                r = f[i+1, j+1]

            if (f[i,j] >= q) and (f[i,j] >= r):
                Z[i,j] = f[i,j]
            else:
                Z[i,j] = 0
    return Z

@jit
def hysteresis(f,high,low):
    N,M = f.shape
    arr_1 = np.zeros((N,M))
    arr_2 = np.zeros((N,M)) 

    for i in range(N-1):
        for j in range(M-1):
            if f[i,j] >= high :
                arr_1[i,j] = True

            elif f[i,j] < high and f[i,j] > low:
                arr_2[i,j] = True

            else:
                arr_1[i,j] = False
                arr_2[i,j] = False
                

    for i in range(1, N-1):
        for j in range(1, M-1):
            if (arr_1[i,j]):
                arr_1[i,j] = 255

            elif((arr_1[i-1,j-1]) or (arr_1[i-1,j]) or (arr_1[i-1,j+1]) or
                (arr_1[i,j-1]) or (arr_1[i,j+1]) or (arr_1[i+1,j-1])
                or (arr_1[i+1,j]) or (arr_1[i+1,j+1])) and (arr_2[i,j]):
                    arr_1[i,j] = 255
            else:
                arr_1[i,j] = 0

    return arr_1

if __name__ == "__main__":
    main()