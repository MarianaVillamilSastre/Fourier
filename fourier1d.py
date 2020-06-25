import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, rfftfreq


funcion=np.genfromtxt("funcion.dat",delimiter="",skip_header=1)

def Transformada_Fourier(datos):
	N=len(datos)
	n=np.arange(0,N)
	transformada=np.zeros(N, dtype=complex)
	for k in range(transformada.shape[0]):
		transformada[k]=np.sum(datos*np.exp(-1j*2*n*np.pi*k/N))
	return transformada
			


print (Transformada_Fourier(funcion[:,1]))

trans=Transformada_Fourier(funcion[:,1])
transformada_final=fftfreq(len(funcion[:,1]))
plt.plot(transformada_final,trans)
plt.show()

tiempo=funcion[:,0]
dt=tiempo[1]
sample_rate=1/dt

frecuencia= np.abs(transformada_final * sample_rate)
maximo=np.max(frecuencia)
print ('La frecuencia es:',maximo,'Hz')


freq =rfftfreq(len(funcion[:,1]),1/dt)
resta=freq[1]-freq[0]
print resta


