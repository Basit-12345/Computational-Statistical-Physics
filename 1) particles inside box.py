import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 1000  # Total number of steps
num_particles = 50  # Number of particles
step_length = 1.0  # Length of each step
box_size = 20.0  # Box side length (centered at 0)
subset_size = 4  # Number of particles to highlight

# Initial positions (randomly chosen within the box)
initial_x = np.random.uniform(-box_size / 2, box_size / 2, size=num_particles)
initial_y = np.random.uniform(-box_size / 2, box_size / 2, size=num_particles)

# Initialize position arrays
x_positions = np.zeros((num_particles, num_steps))
y_positions = np.zeros((num_particles, num_steps))

# Set initial positions
x_positions[:, 0] = initial_x
y_positions[:, 0] = initial_y

# Simulate the random walk with reflective boundary conditions
for step in range(1, num_steps):
    angles = np.random.uniform(0, 2 * np.pi, size=num_particles)  # Random angles
    dx = step_length * np.cos(angles)  # Displacement in x
    dy = step_length * np.sin(angles)  # Displacement in y
    
    # Update positions
    x_positions[:, step] = x_positions[:, step - 1] + dx
    y_positions[:, step] = y_positions[:, step - 1] + dy
    
    # Reflective boundary conditions for x
    x_positions[:, step] = np.where(
        x_positions[:, step] > box_size / 2, 
        box_size - x_positions[:, step], 
        x_positions[:, step]
    )
    x_positions[:, step] = np.where(
        x_positions[:, step] < -box_size / 2, 
        -box_size - x_positions[:, step], 
        x_positions[:, step]
    )
    
    # Reflective boundary conditions for y
    y_positions[:, step] = np.where(
        y_positions[:, step] > box_size / 2, 
        box_size - y_positions[:, step], 
        y_positions[:, step]
    )
    y_positions[:, step] = np.where(
        y_positions[:, step] < -box_size / 2, 
        -box_size - y_positions[:, step], 
        y_positions[:, step]
    )

# Select a random subset of particles to plot
subset_indices = np.random.choice(num_particles, subset_size, replace=False)

# Plot the trajectories of the selected particles
plt.figure(figsize=(10, 8))
for i in subset_indices:
    plt.plot(x_positions[i, :], y_positions[i, :], label=f"Particle {i+1}", alpha=0.7)

# Highlight initial positions of the selected particles
plt.scatter(initial_x[subset_indices], initial_y[subset_indices], 
            color='blue', label='Initial Positions', s=100, edgecolors='black')

# Highlight final positions of the selected particles
final_x = x_positions[subset_indices, -1]
final_y = y_positions[subset_indices, -1]
plt.scatter(final_x, final_y, color='red', label='Final Positions', s=100, edgecolors='black')

# Draw the boundary of the box
plt.axhline(box_size / 2, color="black", linestyle="--", linewidth=1)
plt.axhline(-box_size / 2, color="black", linestyle="--", linewidth=1)
plt.axvline(box_size / 2, color="black", linestyle="--", linewidth=1)
plt.axvline(-box_size / 2, color="black", linestyle="--", linewidth=1)

# Customize plot
plt.xlabel("X Position", fontsize=14)
plt.ylabel("Y Position", fontsize=14)
plt.title("2D Random Walk with Reflective Boundary Conditions", fontsize=14)
plt.grid(True)
plt.legend(loc="upper left", fontsize=10, frameon=True)
plt.tight_layout()
plt.show()
