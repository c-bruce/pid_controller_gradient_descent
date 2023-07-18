import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc, rcParams

from pid_controller import PIDController
from car_simulation import Car
from gradient_descent import GradientDescent, car_cost_function

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

a = np.array([5.0, 1.0, 0.0])
gradient_descent = GradientDescent(a, 0.1, car_cost_function, a_min=[0,0,0])
gradient_descent.execute_adagrad(2000)

mass = 1000.0  # Mass of the car [kg]
Cd = 0.2  # Drag coefficient []
Crr = 0.02 # Rolling resistance []
A = 2.5 # Frontal area of the car [m^2]
Fp = 30 # [N/%]

# Simulation parameters
dt = 0.1  # Time step
total_time = 60.0  # Total simulation time
nsteps = int(total_time / dt)
initial_velocity = 0.0  # Initial velocity of the car [m/s]
target_velocity = 20.0 # Target velocity of the car [m/s]

velocity_results = []
for point in gradient_descent.points:
    pid_controller = PIDController(point[0], point[1], point[2])
    car = Car(mass, Crr, Cd, A, Fp)
    pedal_s, velocity_s, time = car.simulate(nsteps, dt, initial_velocity, target_velocity, pid_controller)
    velocity_results.append(velocity_s)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.axhline(y=target_velocity, color='k', linestyle='--', label='Setpoint $r(t)$', lw=0.5)
ax1_line, = ax1.plot([], [], '-', color='b', label='Process variable $y(t)$')
ax1.text(12.5,2,'$K_p$=5.00, $K_i$=1.00, $K_d$=0.00')
ax1.set_xlabel('Time $[s]$')
ax1.set_ylabel('Velocity $[ms^{-1}]$')
ax1.set_xlim(0, 60)
ax1.set_ylim(0, 30)
ax2_line, = ax2.plot([], [], '-', color='magenta', label='Error magnitude $|e(t)|$')
ax2.set_xlabel('Time $[s]$')
ax2.set_ylabel('Velocity $[ms^{-1}]$')
ax2.set_xlim(0, 60)
ax2.set_ylim(0, 30)
ax3_line, = ax3.plot([], [], '-', color='k', label='Cost $f(a)$')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Cost $[m]$')
ax3.set_xlim(0, 500)
ax3.set_ylim(0, 200)

ax1.legend()
ax2.legend()
ax3.legend()

fig.set_figheight(4)
fig.set_figwidth(15)

iteration_title = fig.suptitle("Iteration: 0")

def update(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.axhline(y=target_velocity, color='k', linestyle='--', label='Setpoint $r(t)$', lw=0.5)
    ax1.plot(time, velocity_results[0:505:5][frame], '-', color='b', label='Process variable $y(t)$')
    Kp = round(gradient_descent.points[0:505:5][frame][0],2)
    Ki = round(gradient_descent.points[0:505:5][frame][1],2)
    Kd = round(gradient_descent.points[0:505:5][frame][2],2)
    ax1.text(12.5,2,f'$K_p$={"{:.2f}".format(Kp)}, $K_i$={"{:.2f}".format(Ki)}, $K_d$={"{:.2f}".format(Kd)}')
    ax1.set_xlabel('Time $[s]$')
    ax1.set_ylabel('Velocity $[ms^{-1}]$')
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 30)
    ax2.plot(time, np.absolute(target_velocity - velocity_results[0:505:5][frame]), '-', color='magenta', label='Error magnitude $|e(t)|$')
    ax2.set_xlabel('Time $[s]$')
    ax2.set_ylabel('Velocity $[ms^{-1}]$')
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 30)
    ax2.fill_between(x=time, 
                     y1=np.absolute(target_velocity - velocity_results[0:505:5][frame]), 
                     where= (0<time)&(time<60),
                     color= "magenta",
                     alpha= 0.2)
    ax3.plot(np.arange(0,5*frame,5), gradient_descent.result[0:505:5][:frame], '-', color='k', label='Cost')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost $[m]$')
    ax3.set_xlim(0, 500)
    ax3.set_ylim(80, 180)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    iteration_title.set_text(f"Iteration: {5*frame}")

anim = FuncAnimation(fig, update, frames=len(velocity_results[0:505:5]), interval=100, blit=False)

anim.save('animated_car_response_plot.gif', writer='pillow')

plt.show()