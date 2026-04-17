#code to obtain the probablity distribution of the final position.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
num_walks = 1000  # Number of random walks
num_steps = 8000  # Number of steps in each walk
d = 5  # Step difference
start = 100  # Starting position

# Simulating multiple random walks
# Each step is either +d or -d with equal probability
walks = np.random.choice([-d, d], size=(num_walks, num_steps))

# Compute the final positions (sum of steps in each walk) and add the start position
final_positions = np.sum(walks, axis=1) + start  # Adding start position

# Theoretical parameters for normal distribution
mean_theoretical = start  # Mean is at the start position
std_theoretical = np.sqrt(num_steps) * d  # Standard deviation considering step size

# Plotting the probability distribution (histogram of final positions)
plt.hist(final_positions, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Simulated Data')

# Generate x values for the theoretical normal distribution
x = np.linspace(min(final_positions), max(final_positions), 1000)
pdf_theoretical = norm.pdf(x, loc=mean_theoretical, scale=std_theoretical)

# Plot the theoretical bell curve
plt.plot(x, pdf_theoretical, color='red', label='Theoretical Normal PDF')

# Add labels, title, and legend
plt.title('Probability Distribution of Final Positions of Random Walks')
plt.xlabel('Final Position')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
