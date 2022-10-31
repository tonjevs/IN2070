from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

filename = 'mona.png'
f = imread(filename, as_gray = True)
plt.imshow(f,cmap='gray')

N,M = f.shape
f_out = np.zeros((N,M))
for i in range(N):
 for j in range(M):
  f_out[i,j] = 1.5*f[i,j]

plt.figure()
plt.imshow(f_out,cmap='gray',vmin=0,vmax=255)
plt.title('f_out')

plt.show()

