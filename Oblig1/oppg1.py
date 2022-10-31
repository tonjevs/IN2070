from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

filename = 'portrett.png'
filename_org = 'portrett.png'
filename2 = 'geometrimaske.png'
f = imread(filename,as_gray = True)
v = imread(filename_org,as_gray = True)
h = imread(filename2,as_gray = True)

sigma = np.std(f)
my = np.mean(f)

my_t = 64
sigma_t = 127

a_temp = (sigma_t**2)/sigma**2
a = np.sqrt(a_temp)
b = my_t - a*my

N,M = f.shape
f_out = np.zeros((N,M))

for i in range(N):
 for j in range(M):
    f_out[i,j] = a*f[i,j] + b

plt.imshow(f_out,cmap='gray')
plt.title('Greytone transformation')
plt.show()

N,M = f_out.shape

g = np.zeros((600,512))
m = np.zeros((600,512))
o = np.zeros((600,512))

A = np.array([[88,84,1,0,0,0],
              [0,0,0,88,84,1],
              [67,120,1,0,0,0],
              [0,0,0,67,120,1],
              [109,129,1,0,0,0],
              [0,0,0,109,129,1]])

b = np.array([258,168,258,342,442,256])

sol = np.linalg.solve(A,b)

T_a = np.array([[sol[0],sol[1],sol[2]],
               [sol[3], sol[4], sol[5]],
               [0,0,1]])

N,M = 600,512
for x in range(N):
    for y in range(M):
        x_mark = round(sol[0]*x+sol[1]*y+sol[2])
        y_mark = round(sol[3]*x+sol[4]*y+sol[5])
        x_mark = int(x_mark)
        y_mark = int(y_mark)
        if x_mark in range(N) and y_mark in range(M):
            o[x_mark,y_mark] = f_out[x,y]

plt.subplot(1,2,1)
plt.imshow(o,cmap='gray')
plt.title('Forward mapping')
plt.subplot(1,2,2)
plt.imshow(h,cmap='gray')
plt.show()

print(np.linalg.inv(T_a))

for x in range(N):
    for y in range(M):
        vector = np.array([x,y,1])
        val = np.dot(np.linalg.inv(T_a), vector)
        g[x,y] = f_out[int(val[0]),int(val[1])]

plt.subplot(1,2,1)
plt.imshow(g,cmap='gray')
plt.title('Backwards mapping nearest neigbour')
plt.subplot(1,2,2)
plt.imshow(h,cmap='gray')
plt.show()


T_a = np.linalg.inv(T_a)
T = np.array([[T_a[0,0],T_a[0,1],T_a[0,2]],
              [T_a[1,0], T_a[1,1], T_a[1,2]],
              [0,0,1]])

for x in range(N):
    for y in range(M):
        i = T[0,0]*x + T[0,1]*y + T[0,2]
        j = T[1,0]*x + T[1,1]*y + T[1,2]
        x0 = min(math.floor(i),0)
        y0 = min(math.floor(j),0)
        x1 = min(math.ceil(i),N-1)
        y1 = min(math.ceil(j),M-1)
        Δx = x - x0
        Δy = y - y0

        p = f_out[x0,y0]+(f_out[x1,y0]-f_out[x0,y0])*Δx
        q = f_out[x0,y1]+(f_out[x1,y1]-f_out[x0,y1])*Δx
        m[x,y] = p+(q-p)*Δy

plt.subplot(1,2,1)
plt.imshow(m,cmap='gray')
plt.title('Backwards mapping bilinear interpolation')
plt.subplot(1,2,2)
plt.imshow(h,cmap='gray')
plt.show()