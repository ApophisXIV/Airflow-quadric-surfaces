import matplotlib.pyplot as plt
import numpy as np


def silla_de_montura(x, y):
    return ((y**2)/5 - (x**2)/3)

# Some example data to display


x = np.arange(-3, 3, 0.8)
y = np.arange(-3, 3, 0.8)
X, Y = np.meshgrid(x, y)
Z = silla_de_montura(X, Y)


fig = plt.figure(figsize=(7, 3))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

ax2 = fig.add_subplot(133, projection='3d')
ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

ax3 = fig.add_subplot(131, projection='3d')
ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

ax4 = fig.add_subplot(133, projection='3d')
ax4.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.show()

# fig, axs = plt.subplots(8, 2 , figsize=(10, 10), sharex=True, sharey=True)

# #Plot 3D surface
# axs[0, 0].plot_surface(x, y, y, cmap='viridis', lw=0.5)


# plt.show()
