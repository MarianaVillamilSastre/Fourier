import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import fftpack



barcelona= plt.imread('Barcelona.jpg')
paris=plt.imread('Paris.jpg')
fractal=plt.imread('frac.jpeg')
triangulos=plt.imread('triangulos.png')

plt.figure()

plt.subplot(221)
plt.imshow(barcelona,cmap= 'Greys_r')

plt.subplot(222)
plt.imshow(paris, cmap= 'Greys_r')

plt.subplot(223)
plt.imshow(fractal,cmap= 'Greys_r')

plt.subplot(224)
plt.imshow(triangulos, cmap='Greys_r')

plt.savefig("imagenes.pdf")

#EL SIGUIENTE METODO ES PARA PASAR LA IMAGEN DE 3D A 2D FUE SACADO DEL BLOG :https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


transformada_barcelona=fftpack.fft2(rgb2gray(barcelona))
transformada_paris=fftpack.fft2(rgb2gray(paris))
transformada_fractal=fftpack.fft2(rgb2gray(fractal))
transformada_triangulos=fftpack.fft2(rgb2gray(triangulos))


plt.figure()

plt.subplot(221)
plt.imshow(np.log10(np.abs(transformada_barcelona**2)))
plt.subplot(222)
plt.imshow(np.log10(np.abs(transformada_paris**2)))

plt.subplot(223)
plt.imshow(np.log10(np.abs(transformada_fractal**2)))

plt.subplot(224)
plt.imshow(np.log10(np.abs(transformada_triangulos**2)))

plt.show()


barcelona_1=transformada_barcelona[405]
paris_1=transformada_paris[509]
fractal_1=transformada_fractal[270]
triangulos_1=transformada_triangulos[286]


plt.figure()

plt.subplot(221)
plt.plot(barcelona_1)

plt.subplot(222)
plt.plot(paris_1)

plt.subplot(223)
plt.plot(fractal_1)

plt.subplot(224)
plt.plot(triangulos_1)

plt.savefig("cortes_transversales.pdf")










