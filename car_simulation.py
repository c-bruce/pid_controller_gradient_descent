import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from pid_controller import PIDController

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

class Car:
    def __init__(self, mass, Crr, Cd, A, Fp):
        self.mass = mass # [kg]
        self.Crr = Crr # [-]
        self.Cd = Cd # [-]
        self.A = A # [m^2]
        self.Fp = Fp # [N/%]
    
    def get_acceleration(self, pedal, velocity):
        # Constants
        rho = 1.225 # [kg/m^3]
        g = 9.81 # [m/s^2]

        # Driving force
        driving_force = self.Fp * pedal

        # Rolling resistance force
        rolling_resistance_force = self.Crr * (self.mass * g)

        # Drag force
        drag_force = 0.5 * rho * (velocity ** 2) * self.Cd * self.A

        acceleration = (driving_force - rolling_resistance_force - drag_force) / self.mass
        return acceleration
    
    def simulate(self, nsteps, dt, velocity, setpoint, pid_controller):
        pedal_s = np.zeros(nsteps)
        velocity_s = np.zeros(nsteps)
        time = np.zeros(nsteps)
        velocity_s[0] = velocity

        for i in range(nsteps - 1):
            pedal = pid_controller.compute(setpoint, velocity, dt)
            pedal = np.clip(pedal, -50, 100)
            pedal_s[i] = pedal

            acceleration = self.get_acceleration(pedal, velocity)

            velocity = velocity_s[i] + acceleration * dt
            velocity_s[i+1] = velocity

            time[i+1] = time[i] + dt
        
        return pedal_s, velocity_s, time

# Simulation parameters
dt = 0.01  # Time step
target_velocity = 20.0  # Velocity set point
total_time = 60.0  # Total simulation time
nsteps = int(total_time / dt)

# Car parameters
mass = 1000.0  # Mass of the car (kg)
rolling_resistance_coeff = 0.01  # Rolling resistance coefficient
Cd = 0.2  # Aerodynamic drag coefficient
Crr = 0.02
A = 2.5 # Frontal area of the car (m^2)
Fp = 30
initial_velocity = 0.0  # Initial velocity of the car

# pid_controller = PIDController(11.07, 0.12, 0.0)
pid_controller = PIDController(5.0, 1.0, 0.0)

car = Car(mass, Crr, Cd, A, Fp)

pedal_s, velocity_s, time = car.simulate(nsteps, dt, initial_velocity, target_velocity, pid_controller)

# Plot results
plt.figure()
plt.plot(time, velocity_s, color='b', label='Process variable $y(t)$')
plt.axhline(y=target_velocity, color='k', linestyle='--', label='Setpoint $r(t)$', lw=0.5)
plt.xlim(0, 60)
plt.ylim(0, 30)
plt.text(30,2,'$K_{p}=5.00, K_{i}=1.00, K_{d}=0.00$')
plt.xlabel('Time $[s]$')
plt.ylabel('Velocity $[ms^{-1}]$')
plt.legend()

plt.figure()
plt.plot(time, np.absolute(target_velocity - velocity_s), color='magenta', label='Error magnitude $|e(t)|$')
plt.fill_between(
        x=time, 
        y1=np.absolute(target_velocity - velocity_s), 
        where= (0<time)&(time<300),
        color= "magenta",
        alpha= 0.2)
plt.xlim(0, 60)
plt.ylim(0, 30)
plt.xlabel('Time $[s]$')
plt.ylabel('Velocity $[ms^{-1}]$')
plt.legend()

plt.show()