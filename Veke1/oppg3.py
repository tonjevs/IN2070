from imageio import imread
import matplotlib.pyplot as plt

bit = 8
f = imread('lena.png')
colormap(gray(256))
f_requantized = floor(double(f)/(2^(8-bit)))
imagesc(f_requantized, [0 ,2^bit-1])