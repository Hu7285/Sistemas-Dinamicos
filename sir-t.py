# Program 04a: Holling-Tanner model. See Figures 4.5 and 4.6.
# Time series and phase portrait for a predator-prey system.
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

b=0.5
g=0.4
N=1000

# Modelo SIR.
def sir(X, t=0):
    # donde X[0] = S(t), X[1] = I(t), X[2] = R(t)
    return np.array([-(b*X[0]*X[1])/N, (b*X[0]*X[1])/N-g*X[1], g*X[1]])

t = np.linspace(0, 200, 1000)
# initial values: S0 = N-1, I0 = 1, R0=0
Sys0 = np.array([N-1, 1, 0])

X, infodict = integrate.odeint(sir, Sys0, t, full_output=True)
x, y, z = X.T

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.5, hspace=0.3)
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(t,x, 'b-', label='S(t)')
ax1.plot(t,y, 'r-', label='I(t)')
ax1.plot(t,z, 'g-', label='R(t)')
ax1.set_title('Time Series')

ax1.set_xlabel('time')
ax1.grid()
ax1.legend(loc='best')

plt.show()