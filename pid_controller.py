import numpy as np
from sim_class import Simulation

class PIDController:
    def __init__(self, kp, ki, kd, target_position, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_position = target_position  # Target position in X, Y
        self.dt = dt  # Time step

        # Error terms
        self.prev_error = np.array([0.0, 0.0])
        self.integral = np.array([0.0, 0.0])

    def compute(self, current_position):
        error = self.target_position[:2] - current_position[:2]  # Only X and Y
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # PID formula
        control_signal = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        return control_signal


# Initialize the simulation
sim = Simulation(num_agents=1, render=True)

# Set working envelope
low = np.array([-0.187, -0.1705, 0.1687], dtype=np.float32)
high = np.array([0.253, 0.2195, 0.2896], dtype=np.float32)

# Random goal position
goal_position = np.random.uniform(low, high)

# PID Controller initialization
pid = PIDController(kp=1.0, ki=0.1, kd=0.05, target_position=goal_position)

# Reset simulation to initial position
initial_position = np.random.uniform(low, high)
sim.reset(num_agents=1)
x, y, z = initial_position
sim.set_start_position(x, y, z)

# Simulation loop
max_steps = 1000
for step in range(max_steps):
    # Get current pipette position
    robot_id = sim.robotIds[0]
    current_position = sim.get_pipette_position(robot_id)

    # Compute PID control action
    action = pid.compute(current_position)

    # Add Z as a placeholder (keep Z fixed), and append an additional flag
    action = np.append(action, [0.0, 0.0])  # Now it has 4 elements

    # Apply action in simulation
    sim.run([action])

    # Check distance to goal
    distance = np.linalg.norm(current_position[:2] - goal_position[:2])  # Only X and Y
    print(f"Step {step}: Distance to goal: {distance:.4f}")

    if distance < 0.001:  # Within 0.1 cm
        print("Goal reached!")
        break


sim.close()
