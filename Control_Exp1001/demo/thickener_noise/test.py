from matplotlib import pyplot as plt

import numpy as np

x,y = np.meshgrid(
    np.linspace(0,1,10),
    np.linspace(0,1,10)
)

plt.contourf(x, y, x+y,  16, alpha=.75, cmap='jet')
plt.colorbar()
plt.scatter([0.4,0.1],[0.4,0.8], c='k',s=5)
plt.legend(['11','22'])
plt.show()
