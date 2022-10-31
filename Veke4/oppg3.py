from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

filename = 'car.png'
f = imread(filename,as_gray = True).astype(int)
plt.imshow(f,cmap='gray')
plt.show()
plt.title('f')
G = 255

h = [0] * 256
p = [0] * 256
c = [0] * 256
T = [0] * 256

for i in f:
 for j in i:
    h[j] += 1

for i in range(G):
    p[i] = h[i]/256

c[0] = p[0]

for i in range(G):
    c[i] = c[i-1] + p[i]

for i in range(G):
    T[i] = round((G-1)*c[i])

for i in f:
    for j in i:
        j = T[j] 

plt.imshow(f,cmap='gray')
plt.hist(f)
plt.show()