# Program 04a: Holling-Tanner model. See Figures 4.5 and 4.6.
# Time series and phase portrait for a predator-prey system.
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

um = 0.1429*7
vm = 1000*7
bh = 6.63e-9
bm = 4.97e-7
g = 1.9e-1
N = 232667

# Modelo SIR.
def sir(X, t=0):
    # donde X[0]=S(t), X[1]=I(t), X[2]=R(t), X[3]=S_m(t), X[4]=I_m(t)
    return np.array([-bh*X[4]*X[0], bh*X[4]*X[0]-g*X[1], g*X[1], vm-bm*X[1]*X[3]-um*X[3], bm*X[1]*X[3]-um*X[4]])

t = np.linspace(0, 25, 1000)
# initial values: S0 = N-1, I0 = 1, R0=0
Sys0 = np.array([N, 1/1741, 0, 3E10, 10000])

X, infodict = integrate.odeint(sir, Sys0, t, full_output=True)
x, y, z, v, w = X.T

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.5, hspace=0.3)
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(t,x, 'b-', label='S_h(t)')
ax1.plot(t,y, 'r-', label='I_h(t)')
ax1.plot(t,z, 'g-', label='R_h(t)')
ax1.set_title('Time Series')

ax1.set_xlabel('time')
ax1.grid()
ax1.legend(loc='best')

plt.show()