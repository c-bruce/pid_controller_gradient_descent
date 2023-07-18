class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_sum = 0
        self.last_error = 0
    
    def compute(self, setpoint, process_variable, dt):
        error = setpoint - process_variable
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self.error_sum += error * dt
        I = self.Ki * self.error_sum
        
        # Derivative term
        D = self.Kd * (error - self.last_error)
        self.last_error = error
        
        # PID output
        output = P + I + D
        
        return output