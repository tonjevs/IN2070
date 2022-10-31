from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

filename = 'mona.png'
f = imread(filename,as_gray = True)
plt.imshow(f,cmap='gray')

plt.show()
plt.title('f')

sigma = np.std(f)
my = np.mean(f)

my_t = 9
sigma_t = 7

a_temp = (sigma_t**2)/sigma**2
a = np.sqrt(a_temp)
b = my_t - a*my

N,M = f.shape
f_out = np.zeros((N,M))
for i in range(N):
 for j in range(M):
    f_out[i,j] = a*f[i,j] + b

plt.title('f_out')
plt.imshow(f_out,cmap='gray')
plt.show()


