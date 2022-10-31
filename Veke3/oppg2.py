from imageio import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

filename = 'mona.png'
f = imread(filename,as_gray = True)
plt.imshow(f,cmap='gray')

plt.figure()
plt.title('f')

N,M = f.shape
f_out = np.zeros((N,M))
for i in range(N):
 for j in range(M):
  f_out[i,j] = f[i,j]

plt.title("Histogram for Image")
plt.xlabel("Value")
plt.ylabel("Pixel frequency")
plt.hist(f_out)
plt.show()

