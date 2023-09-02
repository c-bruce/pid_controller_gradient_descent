import numpy as np

from pid_controller import PIDController
from car_simulation import Car

def quadratic_cost_function(x):
    y = x**2
    return y

def gaussian_cost_function(a):
    x = a[0]
    y = a[1]
    z = np.exp(-((x-0.5)**2 + (y-0.5)**2))-np.exp(-((x+0.5)**2 + (y+0.5)**2))
    return z

def car_cost_function(a):
    # Car parameters
    mass = 1000.0  # Mass of the car [kg]
    Cd = 0.2  # Drag coefficient []
    Crr = 0.02 # Rolling resistance []
    A = 2.5 # Frontal area of the car [m^2]
    Fp = 30 # [N/%]

    # PID controller parameters
    Kp = a[0]
    Ki = a[1]
    Kd = a[2]

    # Simulation parameters
    dt = 0.1  # Time step
    total_time = 60.0  # Total simulation time
    nsteps = int(total_time / dt)
    initial_velocity = 0.0  # Initial velocity of the car [m/s]
    target_velocity = 20.0 # Target velocity of the car [m/s]

    car = Car(mass, Crr, Cd, A, Fp)
    pid_controller = PIDController(Kp, Ki, Kd)

    pedal_s, velocity_s, time = car.simulate(nsteps, dt, initial_velocity, target_velocity, pid_controller)

    cost = np.trapz(np.absolute(target_velocity - velocity_s), time)
    return cost

def get_grad_2d(a, h, func):
    a_h_0 = a + np.array([h, 0])
    a_h_1 = a + np.array([0, h])
    grad = np.array([(func(a_h_0) - func(a)) / h, (func(a_h_1) - func(a)) / h])
    print(grad)
    return grad

def get_grad_3d(a, h, func):
    a_h_0 = a + np.array([h, 0, 0])
    a_h_1 = a + np.array([0, h, 0])
    a_h_2 = a + np.array([0, 0, h])
    grad = np.array([(func(a_h_0) - func(a)) / h, (func(a_h_1) - func(a)) / h, (func(a_h_2) - func(a)) / h])
    # steepest = np.where(np.abs(grad) == max(np.abs(grad)))[0][0]
    # test = np.zeros(3)
    # test[steepest] = 1
    # print(grad * test)
    # print(grad)
    return grad

def get_a(a, learning_rate, grad):
    a = a - (learning_rate * grad)
    print(a)
    return a

class GradientDescent:
    def __init__(self, a, learning_rate, cost_function, a_min=None, a_max=None):
        self.a = a
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.a_min = a_min
        self.a_max = a_max
        self.G = np.zeros([len(a), len(a)])
        self.points = []
        self.result = []
    
    def grad(self, a):
        h = 0.0000001
        a_h = a + (np.eye(len(a)) * h)
        cost_function_at_a = self.cost_function(a)
        grad = []
        for i in range(0, len(a)):
            grad.append((self.cost_function(a_h[i]) - cost_function_at_a) / h)
        grad = np.array(grad)
        return grad
    
    def update_a(self, learning_rate, grad):
        if len(grad) == 1:
            grad = grad[0]
        self.a -= (learning_rate * grad)
        if (self.a_min is not None) or (self.a_max is not None):
            self.a = np.clip(self.a, self.a_min, self.a_max)
    
    def update_G(self, grad):
        self.G += np.outer(grad,grad.T)
    
    def execute(self, iterations):
        for i in range(0, iterations):
            self.points.append(list(self.a))
            self.result.append(self.cost_function(self.a))
            grad = self.grad(self.a)
            self.update_a(self.learning_rate, grad)
    
    def execute_adagrad(self, iterations):
        for i in range(0, iterations):
            self.points.append(list(self.a))
            self.result.append(self.cost_function(self.a))
            grad = self.grad(self.a)
            self.update_G(grad)
            learning_rate = self.learning_rate * np.diag(self.G)**(-0.5)
            self.update_a(learning_rate, grad)
            print(self.a)

# Quadratic
# a = np.array([5.0])
# gradient_descent = GradientDescent(a, 1.02, quadratic_cost_function)
# gradient_descent.execute(10)

# Gaussian
# a = np.array([0.5, 0.5])
# gradient_descent = GradientDescent(a, 0.5, gaussian_cost_function)
# gradient_descent.execute(10)

# Car
# a = np.array([5.0, 1.0, 0.0])
# gradient_descent = GradientDescent(a, 0.1, car_cost_function, a_min=[0,0,0])
# gradient_descent.execute_adagrad(2000)