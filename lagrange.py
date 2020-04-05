import numpy as np
import scipy as sp


x = np.linspace(-1.5, 1.5)

[X,Y] = np.meshgrid(x, x)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_surface( X, Y, X + Y)
theta = np.linspace(0,2*np.pi)
R = 1.0
x1 = R* np.cos(theta)
y1 = R* np.sin(theta)

ax.plot(x1, y1, x1 + y1, "r-")
plt.show()

##contruct the multiplier

def func(X):
    x = X[0]
    y = X[1]
    L = X[2]
    return x + y + L *(x**2 + y**2 - 1)

## partial derivatives
def dfunc(X):
    dLambda = np.zeros(len(X))
    h = le-3 
    for i in range(len(X)):
        dX = np.zeros(len(X))  
        dX[i]  = h
        dLambda[i] = (func(X+dX) -func(X-dX))/(2*h);
    return dLambda

from scipy.optimize import fsolve

#max
X1 = fsolve((dfunc, [1,1,0])
print(X1, func(X1))

#min
X2  =fsolve(dfunc, [-1,-1,0])
print(X2, func(X2))

