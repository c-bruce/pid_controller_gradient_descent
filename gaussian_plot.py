import numpy as np
from mayavi import mlab
from gradient_descent import GradientDescent, gaussian_cost_function


a = np.array([0.8, 0.2])
gradient_descent = GradientDescent(a, 0.1, gaussian_cost_function)
gradient_descent.execute(100)

[x, y] = np.mgrid[-2:2:0.01,-2:2:0.01]
z = np.exp(-((x-0.5)**2 + (y-0.5)**2))-np.exp(-((x+0.5)**2 + (y+0.5)**2))

# View it.
s = mlab.mesh(x, y, z, colormap='coolwarm')
mlab.plot3d(np.array(gradient_descent.points)[:,0], np.array(gradient_descent.points)[:,1], gradient_descent.result)
mlab.show()