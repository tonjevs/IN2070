import math 
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

f = 'geometrimaske.png'
img = imread(f, as_gray=True)
plt.imshow(img, cmap="gray")

N,M = img.shape
f_out = np.zeros((N,M))

#grader = -math.pi/4
grader = 0.575958653
a0 = math.cos(grader)
a1 = -(math.sin(grader))
b0 = math.sin(grader)
b1 = math.cos(grader)

for i in range(N):
    for j in range(M):
        x = round((a0*i)+(a1*j))
        y = round((b0*i)+(b1*j))
        x = int(x)
        y = int(y)
        if x in range(N) and y in range(N):
            f_out[x,y] = img[i,j]

plt.figure()
plt.imshow(f_out,cmap='gray',vmin=0,vmax=255)
plt.show()