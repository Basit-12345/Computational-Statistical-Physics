import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 10000  # Total number of steps
num_particles = 3000  # Number of particles
box_size = 100  # Box side length (centered at 0)
h_value = box_size / 4  # Assume the opening size is box_size / 4
step_lengths = np.linspace(0.1, 5, 50)  # Different step sizes to test

# Function to simulate random walk and calculate half-life
def simulate_with_step_length(step_length):
    initial_x = np.random.uniform(-box_size / 2, box_size / 2, size=num_particles)
    initial_y = np.random.uniform(-box_size / 2, box_size / 2, size=num_particles)

    x_positions = np.zeros((num_particles, num_steps))
    y_positions = np.zeros((num_particles, num_steps))

    x_positions[:, 0] = initial_x
    y_positions[:, 0] = initial_y

    escaped = np.zeros(num_particles, dtype=bool)  # Track escaped particles
    for step in range(1, num_steps):
        angles = np.random.uniform(0, 2 * np.pi, size=num_particles)
        dx = step_length * np.cos(angles)
        dy = step_length * np.sin(angles)

        # Update positions
        x_positions[:, step] = x_positions[:, step - 1] + dx
        y_positions[:, step] = y_positions[:, step - 1] + dy

        # Apply reflective boundary conditions
        x_positions[:, step] = np.where(
            np.abs(x_positions[:, step]) > box_size / 2,
            np.sign(x_positions[:, step]) * (box_size - np.abs(x_positions[:, step])),
            x_positions[:, step],
        )
        y_positions[:, step] = np.where(
            (y_positions[:, step] > box_size / 2) & ~escaped,
            box_size - (y_positions[:, step] - box_size),
            y_positions[:, step],
        )
        y_positions[:, step] = np.where(
            y_positions[:, step] < -box_size / 2,
            -box_size - (y_positions[:, step] + box_size),
            y_positions[:, step],
        )

        # Check for escapes
        escape_condition = (
            (y_positions[:, step] > box_size / 2)
            & (np.abs(x_positions[:, step]) <= h_value / 2)  # Use fixed opening size
        )
        escaped = escaped | escape_condition

        # Stop updating escaped particles
        x_positions[escaped, step] = x_positions[escaped, step - 1]
        y_positions[escaped, step] = y_positions[escaped, step - 1]

        # Calculate half-life
        num_remaining = np.sum(~escaped)
        if num_remaining <= num_particles / 2:
            return step  # Return step where half-life is reached

    return num_steps  # Return total steps if half-life is not reached

# Calculate half-life for each step size
half_lives = [simulate_with_step_length(step_length) for step_length in step_lengths]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(step_lengths, half_lives, marker='o', linestyle='-')
plt.xlabel("Step Length", fontsize=14)
plt.ylabel("Half-Life (Steps)", fontsize=14)
plt.title("Half-Life vs. Step Length", fontsize=14)
plt.grid(True)
plt.show()
