import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

x = np.arange(-7.5, 7.5, 0.01)
y = x ** 2

# gamma = 0.1
x_small_gamma = np.array([5.0, 3.99999999, 3.19999998, 2.55999997, 2.04799997, 1.63839997, 1.31071996, 1.04857596, 0.83886076,
                          0.6710886, 0.53687087])
y_small_gamma = x_small_gamma ** 2

# gamma = 1.01
x_big_gamma = np.array([5.0, -5.20000012, 5.40800007, -5.62432019, 5.84929293, -6.08326478, 6.32659528, -6.57965925,
                        6.84284552, -7.11655946, 7.40122178])
y_big_gamma = x_big_gamma ** 2

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, y, 'k', lw=0.5)
ax1_line, = ax1.plot([], [], '--bo', label='$\gamma=0.1$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax2.plot(x, y, 'k', lw=0.5)
ax2_line, = ax2.plot([], [], '--ro', label='$\gamma=1.02$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')

ax1.legend()
ax2.legend()
ax1.set_aspect(1./ax1.get_data_ratio())
ax2.set_aspect(1./ax2.get_data_ratio())
fig.set_figheight(5)
fig.set_figwidth(10)

iteration_title = fig.suptitle("Iteration: 0")

def update(frame):
    ax1_line.set_data(x_small_gamma[:frame], y_small_gamma[:frame])
    ax2_line.set_data(x_big_gamma[:frame], y_big_gamma[:frame])
    iteration_title.set_text(f"Iteration: {frame}")

anim = FuncAnimation(fig, update, frames=len(x_small_gamma), interval=500, blit=False)

anim.save('quadratic_plot.gif', writer='pillow')

plt.show()