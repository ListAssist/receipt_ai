import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def gauss(x, y, sigma = 1):
    return math.e ** (-(x**2 + y ** 2) / (2 * sigma ** 2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-4, 4, .1)
y = np.arange(-4, 4, .1)
xx, yy = np.meshgrid(x, y)
z = gauss(xx, yy)
print(xx)
surface = ax.plot_surface(xx, yy, z, cmap=cm.seismic, antialiased=False)
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.title("Gaussian Filter function with $\sigma=1$")
plt.show()