import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

B= 0.0022 
gamma= 0.45

def s_prime(I,S,R,t):
	return -B*I*S


def i_prime(I,S,R,t):
	return B*I*S-gamma*I

def r_prime(I,S,R,t):
	return gamma*I



delta=0.0005
puntos=771
tiempo, suceptibles,infectados,recuperados=np.zeros(puntos),np.zeros(puntos),np.zeros(puntos),np.zeros(puntos)

def RungeK(i1,s1,r1,t1):
	k1s=s_prime(i1,s1,r1,t1)
	k1i=i_prime(i1,s1,r1,t1)
	k1r=r_prime(i1,s1,r1,t1)
	



	k2s=s_prime(i1 + k1i*delta*0.5,s1 + k1s*delta*0.5, r1 + k1r*delta*0.5, t1 + delta*0.5)
	k2i=i_prime(i1 + k1i*delta*0.5,s1 + k1s*delta*0.5, r1 + k1r*delta*0.5, t1 + delta*0.5)
	k2r=r_prime(i1 + k1i*delta*0.5,s1 + k1s*delta*0.5, r1 + k1r*delta*0.5, t1 + delta*0.5)
	



	k3s=s_prime(i1 + k2i*delta*0.5,s1 + k2s*delta*0.5, r1 + k2r*delta*0.5, t1 + delta*0.5)
	k3i=i_prime(i1 + k2i*delta*0.5,s1 + k2s*delta*0.5, r1 + k2r*delta*0.5, t1 + delta*0.5)
	k3r=r_prime(i1 + k2i*delta*0.5,s1 + k2s*delta*0.5, r1 + k2r*delta*0.5, t1 + delta*0.5)
	



	k4s=s_prime(i1 + k3i*delta,s1 + k3s*delta, r1 + k3r*delta, t1 + delta)
	k4i=i_prime(i1 + k3i*delta,s1 + k3s*delta, r1 + k3r*delta, t1 + delta)
	k4r=r_prime(i1 + k3i*delta,s1 + k3s*delta, r1 + k3r*delta, t1 + delta)
	
		


	promedio_s_prime=(k1s+2.0*k2s+2.0*k3s+k4s)/6.0
	s2=s1 +promedio_s_prime*delta
	promedio_i_prime=(k1i+2.0*k2i+2.0*k3i+k4i)/6.0
	i2=i1+promedio_i_prime*delta
	promedio_r=(k1r+2.0*k2r+2.0*k3r+k4r)/6.0
	r2=r1+promedio_r*delta

	t2=t1+delta




	return s2,i2,r2,t2


tiempo[0], suceptibles[0],infectados[0],recuperados[0]=0.0,770.0,1.0,0.0


for i in range(1,771):
	infectados[i],suceptibles[i],recuperados[i],tiempo[i]=RungeK(infectados[i-1],suceptibles[i-1],recuperados[i-1],tiempo[i-1])


#Siguiendo el enunciado, se guarda la grafica en un .pdf
plt.plot(tiempo,suceptibles,c="DeepSkyBlue",label="Runge-Kutta")
plt.legend(loc=0)
plt.ylabel("y(m)")
plt.xlabel("x(m)")
plt.show()




