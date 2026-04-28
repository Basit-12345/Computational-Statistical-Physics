import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 10000  # Total number of steps
num_particles = 3000  # Number of particles
step_length = 1.0  # Length of each step
box_size = 100  # Box side length (centered at 0)
subset_size = 4  # Number of particles to highlight
h_values = np.linspace(1, box_size, box_size)  # Opening sizes to test
num_simulations = 10  # Number of times to run the simulation for averaging

# Function to simulate random walk and calculate half-life
def simulate_with_opening(d):
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
            & (np.abs(x_positions[:, step]) <= d / 2)
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

# Run multiple simulations and calculate the average half-life for each opening size
avg_half_lives = []
for h in h_values:
    half_life_values = [simulate_with_opening(h) for _ in range(num_simulations)]
    avg_half_lives.append(np.mean(half_life_values))  # Average over the simulations

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(h_values, avg_half_lives, marker='o', linestyle='-')
plt.xlabel("Opening Size (h)", fontsize=14)
plt.ylabel("Average Half-Life (Steps)", fontsize=14)
plt.title("Average Half-Life vs. Opening Size", fontsize=14)
plt.grid(True)
plt.show()
